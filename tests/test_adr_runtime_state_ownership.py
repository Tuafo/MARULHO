"""Tests for ADR 0002 Runtime State ownership and CONTEXT.md alignment (Issue #32)."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVICE_SRC_ROOT = _REPO_ROOT / "src" / "marulho" / "service"
_ADR_PATH = _REPO_ROOT / "docs" / "adr" / "0002-runtime-state-ownership.md"
_CONTEXT_PATH = _REPO_ROOT / "CONTEXT.md"
_RUNTIME_STATE_FIELDS = ("dirty_state", "state_revision", "last_event", "recent_events")
_RUNTIME_STATE_PRIVATE_FIELDS = ("_dirty_state", "_state_revision", "_brain_last_event", "_brain_event_history")
_RUNTIME_STATE_PRIVATE_FIELD_SET = frozenset(_RUNTIME_STATE_PRIVATE_FIELDS)
_DYNAMIC_ATTRIBUTE_ACCESSORS = frozenset({"getattr", "setattr", "delattr"})


def _runtime_state_private_field_reference(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and node.attr in _RUNTIME_STATE_PRIVATE_FIELD_SET:
        return node.attr
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _RUNTIME_STATE_PRIVATE_FIELD_SET:
        return node.name
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DYNAMIC_ATTRIBUTE_ACCESSORS:
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
            return None
        attr_name = node.args[1].value
        if isinstance(attr_name, str) and attr_name in _RUNTIME_STATE_PRIVATE_FIELD_SET:
            return f"{node.func.id}({attr_name})"
    return None


def _iter_runtime_state_private_field_violations() -> list[str]:
    violations: list[str] = []
    for path in sorted(_SERVICE_SRC_ROOT.rglob("*.py")):
        if path.name == "runtime_state.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            field_reference = _runtime_state_private_field_reference(node)
            if field_reference is not None:
                violations.append(
                    f"{path.relative_to(_REPO_ROOT)}:{node.lineno}:{node.col_offset + 1} {field_reference}"
                )
    return violations


class TestRuntimeStateOwnershipADR(unittest.TestCase):
    """ADR 0002 must exist and describe Runtime State ownership."""

    def test_adr_file_exists(self) -> None:
        self.assertTrue(_ADR_PATH.exists(), f"ADR file not found at {_ADR_PATH}")

    def test_adr_has_status_accepted(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("## Status", text)
        self.assertRegex(text, r"## Status\s*\n\s*Accepted")

    def test_adr_mentions_runtime_state_ownership(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("Runtime State", text)
        self.assertIn("brain event history", text)
        self.assertIn("owns mutation truth", text)

    def test_adr_mentions_previous_service_manager_field_ownership(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("Service Manager", text)
        self.assertIn("shallow", text.lower())
        self.assertIn("implicit", text.lower())
        for field in _RUNTIME_STATE_FIELDS:
            self.assertIn(field, text)

    def test_adr_documents_payload_compatibility_and_replay_exception(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("external payload contract", text.lower())
        self.assertIn("dirty-without-revision", text)
        self.assertIn("mark_dirty_without_revision", text)

    def test_adr_does_not_contradict_adr_0001(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("ADR 0001", text)
        self.assertIn("does not reopen or contradict", text)


class TestContextMdUpdated(unittest.TestCase):
    """CONTEXT.md must match the Runtime State ownership decision."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.context_text = _CONTEXT_PATH.read_text(encoding="utf-8")

    def test_context_mentions_runtime_state_fields(self) -> None:
        self.assertIn("Runtime State", self.context_text)
        for field in _RUNTIME_STATE_FIELDS:
            self.assertIn(field, self.context_text)

    def test_context_removes_interaction_pipeline_revision_ownership(self) -> None:
        self.assertIn("Interaction Pipeline", self.context_text)
        self.assertIn("query gap history", self.context_text)
        self.assertNotIn("dirty-state revision counter", self.context_text)

    def test_context_mentions_dirty_without_revision_replay_path(self) -> None:
        self.assertIn("Replay Controller", self.context_text)
        self.assertIn("dirty-without-revision", self.context_text)
        self.assertIn("state_revision", self.context_text)


class TestRuntimeStateArchitectureGuard(unittest.TestCase):
    """Service runtime modules must not reopen the removed manager compatibility seam."""

    def test_no_service_runtime_module_defines_or_accesses_runtime_state_private_fields(self) -> None:
        violations = _iter_runtime_state_private_field_violations()
        self.assertFalse(
            violations,
            "Found direct runtime-truth private-field definition or access outside RuntimeState:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
