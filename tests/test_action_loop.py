from __future__ import annotations

import json
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from urllib.parse import parse_qs, urlparse

from hecsn.service.action_loop import execute_digital_action, execute_workspace_search


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


class ActionLoopTests(unittest.TestCase):
    def test_workspace_search_verifies_grounded_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )

            result = execute_workspace_search(
                root,
                query_text="cats chase mice",
                predicted_outcome="I expect to find evidence about cats chasing mice.",
            )

            self.assertEqual(result.action_type, "workspace_search")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertGreaterEqual(len(result.verification.evidence), 1)
            self.assertIn("cats chase mice", result.actual_outcome.lower())
            self.assertIn("workspace_search", result.episode_text)
            self.assertEqual(result.memory_metadata()["verification_status"], "verified")

    def test_workspace_search_records_contradiction_when_no_hit_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )

            result = execute_digital_action(
                root,
                {
                    "action_type": "workspace_search",
                    "query_text": "aurora borealis",
                    "predicted_outcome": "I expect to find aurora evidence.",
                },
            )

            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("no matching file hits", result.actual_outcome.lower())
            self.assertEqual(result.memory_metadata()["verification_status"], "contradicted")
            self.assertEqual(result.memory_metadata()["observation_kind"], "action")

    def test_workspace_read_verifies_existing_file_and_extracts_relevant_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )

            result = execute_digital_action(
                root,
                {
                    "action_type": "workspace_read",
                    "path": "notes.md",
                    "query_text": "cats chase night",
                    "predicted_outcome": "I expect notes.md to mention what cats chase at night.",
                },
            )

            self.assertEqual(result.action_type, "workspace_read")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertGreaterEqual(len(result.verification.evidence), 1)
            self.assertEqual(result.verification.evidence[0]["path"], "notes.md")
            self.assertIn("cats chase mice at night", result.actual_outcome.lower())
            self.assertIn("workspace_read", result.episode_text)

    def test_workspace_read_contradicts_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = execute_digital_action(
                root,
                {
                    "action_type": "workspace_read",
                    "path": "missing.md",
                    "query_text": "cats chase night",
                    "predicted_outcome": "I expect to read missing.md.",
                },
            )

            self.assertEqual(result.action_type, "workspace_read")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("could not open", result.actual_outcome.lower())

    def test_web_fetch_verifies_local_http_content(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "web_fetch",
                        "url": f"http://127.0.0.1:{port}/page.html",
                        "query_text": "cats chase night",
                        "predicted_outcome": "I expect the page to mention what cats chase at night.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "web_fetch")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertGreaterEqual(len(result.verification.evidence), 1)
            self.assertIn("cats chase mice at night", result.actual_outcome.lower())
            self.assertIn("http://127.0.0.1", result.verification.evidence[0]["path"])

    def test_web_fetch_contradicts_invalid_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = execute_digital_action(
                root,
                {
                    "action_type": "web_fetch",
                    "url": "not-a-valid-url",
                    "query_text": "cats chase night",
                    "predicted_outcome": "I expect to fetch a web page.",
                },
            )

            self.assertEqual(result.action_type, "web_fetch")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("valid http/https url", result.actual_outcome.lower())

    def test_api_request_verifies_local_json_content(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/data.json",
                        "query_text": "cats chase night",
                        "predicted_outcome": "I expect the JSON endpoint to say what cats chase at night.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertGreaterEqual(len(result.verification.evidence), 1)
            self.assertIn("relevant json fields", result.actual_outcome.lower())
            self.assertIn("$.facts.chase", result.verification.evidence[0]["json_path"])

    def test_api_request_verifies_structured_json_object_summary(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
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

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertGreaterEqual(len(result.verification.evidence), 1)
            self.assertEqual(result.verification.evidence[0]["json_path"], "$.animals[0]")
            self.assertEqual(result.verification.evidence[0]["structure_kind"], "object")
            self.assertGreaterEqual(result.verification.evidence[0]["field_count"], 3)
            self.assertIn("name = cat", result.verification.evidence[0]["snippet"].lower())
            self.assertIn("relevant json structures", result.actual_outcome.lower())

    def test_api_request_verifies_expected_json_paths_and_response_shape(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
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

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertEqual(result.inputs["expected_json_paths"], ["$.animals[0]", "$.animals[0].diet"])
            self.assertEqual(result.inputs["expected_response_shape"], "object")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_path" for item in result.verification.evidence))
            self.assertTrue(any(item.get("assertion_kind") == "expected_response_shape" for item in result.verification.evidence))
            self.assertIn("satisfied", result.actual_outcome.lower())

    def test_api_request_contradicts_missing_expected_json_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "animals.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice", "active_time": "night"},
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/animals.json",
                        "expected_json_paths": ["$.animals[1].diet"],
                        "query_text": "animals",
                        "predicted_outcome": "I expect a second animal diet field.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("expected json paths", result.actual_outcome.lower())
            self.assertIn("$.animals[1].diet", result.actual_outcome)

    def test_api_request_contradicts_response_shape_assertion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "animals.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice", "active_time": "night"},
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/animals.json",
                        "expected_response_shape": "array",
                        "query_text": "animals",
                        "predicted_outcome": "I expect an array response.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("response shape", result.actual_outcome.lower())
            self.assertIn("not expected 'array'", result.actual_outcome.lower())

    def test_api_request_verifies_expected_json_values(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
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

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertEqual(result.inputs["expected_json_values"]["$.animals[0].diet"], "mice")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_value" for item in result.verification.evidence))
            self.assertIn("json value assertion", result.actual_outcome.lower())

    def test_api_request_contradicts_expected_json_value_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "animals.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice", "active_time": "night"},
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/animals.json",
                        "expected_json_values": {"$.animals[0].diet": "grass"},
                        "query_text": "animals",
                        "predicted_outcome": "I expect the first animal diet to be grass.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("instead of expected", result.actual_outcome.lower())
            self.assertEqual(result.verification.evidence[0]["assertion_kind"], "expected_json_value")
            self.assertEqual(result.verification.evidence[0]["actual_value"], "mice")
            self.assertEqual(result.verification.evidence[0]["expected_value"], "grass")

    def test_api_request_verifies_expected_json_predicates(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/metrics.json",
                        "expected_json_predicates": [
                            {"path": "$.animals[0].diet", "op": "contains", "value": "night"},
                            {"path": "$.animals[0].diet", "op": "regex", "value": "mice\\s+at"},
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

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertEqual(result.inputs["expected_json_predicates"][0]["op"], "contains")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate" for item in result.verification.evidence))
            self.assertTrue(any(item.get("predicate_op") == "gte" for item in result.verification.evidence))
            self.assertIn("json predicate assertion", result.actual_outcome.lower())

    def test_api_request_contradicts_expected_json_predicate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {"score": 0.91},
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/metrics.json",
                        "expected_json_predicates": [
                            {"path": "$.metrics.score", "op": "lt", "value": 0.5},
                        ],
                        "query_text": "metrics",
                        "predicted_outcome": "I expect the score to be below one half.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("did not satisfy predicate", result.actual_outcome.lower())
            self.assertEqual(result.verification.evidence[0]["assertion_kind"], "expected_json_predicate")
            self.assertEqual(result.verification.evidence[0]["predicate_op"], "lt")
            self.assertEqual(result.verification.evidence[0]["actual_value"], 0.91)

    def test_api_request_verifies_composite_json_predicates(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/metrics.json",
                        "expected_json_predicates": [
                            {"path": "$.metrics.score", "op": "between", "value": {"min": 0.9, "max": 1.0}},
                            {"path": "$.animals[0].diet", "op": "startswith", "value": "mice"},
                            {"path": "$.animals[0].diet", "op": "endswith", "value": "night"},
                            {"path": "$.tags", "op": "any_contains", "value": "hunter"},
                            {"path": "$.tags", "op": "any_regex", "value": "^feline"},
                        ],
                        "query_text": "cat metrics night",
                        "predicted_outcome": "I expect the JSON endpoint to satisfy composite predicate checks.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertEqual(result.inputs["expected_json_predicates"][0]["op"], "between")
            self.assertTrue(any(item.get("predicate_op") == "between" for item in result.verification.evidence))
            self.assertTrue(any(item.get("predicate_op") == "any_contains" for item in result.verification.evidence))
            self.assertIn("json predicate assertion", result.actual_outcome.lower())

    def test_api_request_contradicts_composite_json_predicate_mismatch(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/metrics.json",
                        "expected_json_predicates": [
                            {"path": "$.metrics.score", "op": "between", "value": {"min": 0.95, "max": 1.0}},
                        ],
                        "query_text": "metrics",
                        "predicted_outcome": "I expect the score to fall inside the higher range.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("did not satisfy predicate", result.actual_outcome.lower())
            self.assertEqual(result.verification.evidence[0]["assertion_kind"], "expected_json_predicate")
            self.assertEqual(result.verification.evidence[0]["predicate_op"], "between")
            self.assertEqual(result.verification.evidence[0]["actual_value"], 0.91)

    def test_api_request_verifies_logical_predicate_groups_and_object_quantifiers(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/logic.json",
                        "expected_json_predicates": [
                            {"path": "$.traits", "op": "all_regex", "value": "^[a-z-]+$"},
                            {"path": "$.traits", "op": "any_contains", "value": "hunter"},
                            {"path": "$.traits", "op": "none_contains", "value": "reptile"},
                        ],
                        "expected_json_predicate_groups": [
                            {
                                "logic": "any",
                                "predicates": [
                                    {"path": "$.metrics.score", "op": "lt", "value": 0.5},
                                    {"path": "$.animals[0].diet", "op": "contains", "value": "night"},
                                ],
                            },
                            {
                                "logic": "none",
                                "predicates": [
                                    {"path": "$.traits", "op": "any_contains", "value": "reptile"},
                                    {"path": "$.animals[0].diet", "op": "contains", "value": "grass"},
                                ],
                            },
                        ],
                        "query_text": "cat logic night",
                        "predicted_outcome": "I expect the JSON endpoint to satisfy logical predicate groups and object quantifiers.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertEqual(result.inputs["expected_json_predicate_groups"][0]["logic"], "any")
            self.assertTrue(any(item.get("predicate_op") == "all_regex" for item in result.verification.evidence))
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate_group" for item in result.verification.evidence))
            self.assertTrue(any(item.get("group_logic") == "any" for item in result.verification.evidence))

    def test_api_request_contradicts_logical_predicate_group_mismatch(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/logic.json",
                        "expected_json_predicate_groups": [
                            {
                                "logic": "all",
                                "predicates": [
                                    {"path": "$.metrics.score", "op": "gt", "value": 0.95},
                                    {"path": "$.animals[0].diet", "op": "contains", "value": "night"},
                                ],
                            }
                        ],
                        "query_text": "logic",
                        "predicted_outcome": "I expect all predicates in the group to hold.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("predicate group", result.actual_outcome.lower())
            self.assertEqual(result.verification.evidence[0]["assertion_kind"], "expected_json_predicate_group")
            self.assertEqual(result.verification.evidence[0]["group_logic"], "all")

    def test_api_request_verifies_wildcard_json_paths_and_values(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/wild.json",
                        "expected_json_paths": ["$.animals[*].diet", "$.groups.predators[*].name"],
                        "expected_json_values": {"$.animals[*].name": "owl"},
                        "expected_json_predicates": [
                            {"path": "$.animals[*].traits[*]", "op": "contains", "value": "night"},
                        ],
                        "query_text": "wild json predators",
                        "predicted_outcome": "I expect wildcard JSON checks to find repeated nested matches.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertTrue(any(item.get("asserted_json_path") == "$.animals[*].diet" for item in result.verification.evidence))
            self.assertTrue(any(item.get("asserted_json_path") == "$.animals[*].name" for item in result.verification.evidence))
            self.assertTrue(any(item.get("wildcard_match") for item in result.verification.evidence))

    def test_api_request_verifies_wildcard_nested_logical_groups(self) -> None:
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/wild.json",
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
                        "query_text": "wild json logic",
                        "predicted_outcome": "I expect wildcard nested logical groups to pass.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertEqual(result.inputs["expected_json_predicate_groups"][0]["logic"], "all")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate_group" for item in result.verification.evidence))
            self.assertTrue(any(item.get("group_logic") == "all" for item in result.verification.evidence))

    def test_api_request_contradicts_wildcard_nested_logical_group_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "wild.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice at night", "traits": ["hunter", "feline"]},
                            {"name": "owl", "diet": "mice at dawn", "traits": ["bird", "night"]},
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
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/wild.json",
                        "expected_json_predicate_groups": [
                            {
                                "logic": "all",
                                "groups": [
                                    {
                                        "logic": "all",
                                        "predicates": [
                                            {"path": "$.animals[*].diet", "op": "contains", "value": "night"},
                                            {"path": "$.animals[*].traits[*]", "op": "contains", "value": "reptile"},
                                        ],
                                    }
                                ],
                            }
                        ],
                        "query_text": "wild json logic",
                        "predicted_outcome": "I expect wildcard nested logical groups to require every child condition.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("predicate group", result.actual_outcome.lower())
            self.assertEqual(result.verification.evidence[0]["assertion_kind"], "expected_json_predicate_group")
            self.assertEqual(result.verification.evidence[0]["group_logic"], "all")

    def test_api_request_supports_parameterized_post_json_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            port = _free_port()
            server = ThreadingHTTPServer(("127.0.0.1", port), _EchoJsonApiHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/api/echo",
                        "method": "POST",
                        "params": {"kind": "feline"},
                        "json_body": {"topic": "cats", "fact": "mice at night"},
                        "query_text": "cats mice night feline",
                        "predicted_outcome": "I expect the API request to return structured JSON about cats and mice at night.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertTrue(result.verification.success)
            self.assertEqual(result.verification.status, "verified")
            self.assertEqual(result.inputs["method"], "POST")
            self.assertEqual(result.inputs["params"]["kind"], "feline")
            self.assertEqual(result.inputs["json_body"]["topic"], "cats")
            self.assertIn("/api/echo?kind=feline", result.inputs["url"])
            self.assertTrue(any(item.get("json_path") == "$.payload.fact" for item in result.verification.evidence))

    def test_api_request_contradicts_get_json_body_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = execute_digital_action(
                root,
                {
                    "action_type": "api_request",
                    "url": "http://127.0.0.1/example.json",
                    "method": "GET",
                    "json_body": {"topic": "cats"},
                    "query_text": "cats",
                    "predicted_outcome": "I expect JSON.",
                },
            )

            self.assertEqual(result.action_type, "api_request")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("json body with get", result.actual_outcome.lower())

    def test_api_request_contradicts_non_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "page.html").write_text(
                "<html><body><main><p>Cats chase mice at night.</p></main></body></html>",
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = execute_digital_action(
                    root,
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/page.html",
                        "query_text": "cats chase night",
                        "predicted_outcome": "I expect JSON.",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(result.action_type, "api_request")
            self.assertFalse(result.verification.success)
            self.assertTrue(result.verification.contradiction)
            self.assertEqual(result.verification.status, "contradicted")
            self.assertIn("valid json", result.actual_outcome.lower())
