"""Tests for ADR 0001 and CONTEXT.md Living Loop deepening documentation (Issue #8).

Verifies that the architecture decision record, context document, and module
docstrings satisfy the acceptance criteria for the Living Loop deepening split.
"""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ADR_PATH = _REPO_ROOT / "docs" / "adr" / "0001-living-loop-depth-aligned-module-split.md"
_CONTEXT_PATH = _REPO_ROOT / "CONTEXT.md"

_DEPTH_LAYER_TERMS = {
    "living_loop_helpers": ["Foundation", "Layer 0"],
    "living_loop_records": ["Layer A"],
    "living_loop_policy": ["Layer B"],
    "living_loop_replay": ["Layer C"],
    "living_loop_self_model": ["Layer D"],
}


class TestADRExists(unittest.TestCase):
    """ADR 0001 must exist and document the four-layer depth stack decision."""

    def test_adr_file_exists(self):
        self.assertTrue(_ADR_PATH.exists(), f"ADR file not found at {_ADR_PATH}")

    def test_adr_has_status_accepted(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("## Status", text)
        self.assertRegex(text, r"## Status\s*\n\s*Accepted")

    def test_adr_documents_four_layers(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        for layer_label in ["Runtime Records", "Policy Scoring", "Replay Planning", "Operational Self-Model"]:
            self.assertIn(layer_label, text, f"ADR missing layer: {layer_label}")

    def test_adr_documents_helpers_module(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("Shared Helpers", text)
        self.assertIn("living_loop_helpers", text)

    def test_adr_documents_depth_stack_table(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        # The decision section should have a table with depth layers
        self.assertIn("Layer A", text)
        self.assertIn("Layer B", text)
        self.assertIn("Layer C", text)
        self.assertIn("Layer D", text)

    def test_adr_documents_dependency_direction(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("unidirectional", text)
        # Must show the dependency chain
        self.assertIn("Helpers", text)
        self.assertIn("Records", text)
        self.assertIn("Policy", text)
        self.assertIn("Replay", text)
        self.assertIn("Self-Model", text)


class TestADRDependencyConstraintAndImportStrategy(unittest.TestCase):
    """ADR must explain the unidirectional dependency constraint and direct-import strategy."""

    def test_adr_explains_unidirectional_constraint(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("Unidirectional dependency constraint", text)
        self.assertIn("no module may import from a higher layer", text.lower())

    def test_adr_explains_direct_import_strategy(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("Direct-import enforcement strategy", text)
        self.assertIn("direct imports", text.lower())

    def test_adr_explains_consumer_migration(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertTrue(
            "consumer" in text.lower() and "direct import" in text.lower(),
            "ADR must explain that consumers import owning modules directly",
        )

    def test_adr_documents_no_aggregator_exports(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertTrue(
            "aggregator" in text.lower() and "__all__" in text,
            "ADR must document that aggregator exports are no longer the active surface",
        )


class TestADRRationale(unittest.TestCase):
    """ADR must record the rationale for four modules, direct imports, and helpers module."""

    def test_adr_explains_why_four_modules(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("Why five modules", text)
        self.assertIn("two-module", text)
        self.assertIn("three-module", text)

    def test_adr_explains_why_direct_imports(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("Why direct imports", text)

    def test_adr_explains_why_helpers_module(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("Why a shared helpers module", text)

    def test_adr_has_consequences_section(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("## Consequences", text)
        self.assertIn("### Positive", text)
        self.assertIn("### Negative", text)

    def test_adr_references_prd(self):
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("PRD #1", text)


class TestContextMdUpdated(unittest.TestCase):
    """CONTEXT.md must include module-level entries for each new module."""

    @classmethod
    def setUpClass(cls):
        cls.context_text = _CONTEXT_PATH.read_text(encoding="utf-8")

    def test_context_mentions_adr_0001(self):
        self.assertIn("ADR 0001", self.context_text)

    def test_context_mentions_helpers_module(self):
        self.assertIn("living_loop_helpers", self.context_text)

    def test_context_mentions_records_module(self):
        self.assertIn("living_loop_records", self.context_text)

    def test_context_mentions_policy_module(self):
        self.assertIn("living_loop_policy", self.context_text)

    def test_context_mentions_replay_module(self):
        self.assertIn("living_loop_replay", self.context_text)

    def test_context_mentions_self_model_module(self):
        self.assertIn("living_loop_self_model", self.context_text)

    def test_context_mentions_deleted_shim(self):
        self.assertIn("compatibility shim is deleted", self.context_text)

    def test_context_maps_records_to_domain_vocab(self):
        self.assertIn("Living Loop records", self.context_text)

    def test_context_maps_policy_to_domain_vocab(self):
        self.assertIn("Policy Actuator", self.context_text)

    def test_context_maps_replay_to_domain_vocab(self):
        self.assertIn("Replay Pipeline planning stage", self.context_text)

    def test_context_maps_self_model_to_domain_vocab(self):
        self.assertIn("Runtime Truth", self.context_text)

    def test_context_documents_layer_labels(self):
        for module_name, terms in _DEPTH_LAYER_TERMS.items():
            for term in terms:
                self.assertIn(
                    term,
                    self.context_text,
                    f"CONTEXT.md missing layer term '{term}' for {module_name}",
                )


class TestModuleDocstrings(unittest.TestCase):
    """Each new module must have a docstring explaining its depth layer and dependencies."""

    def _get_docstring(self, module_name: str) -> str:
        mod = importlib.import_module(module_name)
        self.assertIsNotNone(mod.__doc__, f"{module_name} has no module docstring")
        return mod.__doc__  # type: ignore[return-value]

    def test_helpers_docstring_mentions_cross_layer(self):
        doc = self._get_docstring("hecsn.service.living_loop_helpers")
        self.assertIn("cross-layer", doc.lower())

    def test_helpers_docstring_mentions_no_upward_dependency(self):
        doc = self._get_docstring("hecsn.service.living_loop_helpers")
        self.assertTrue(
            "no upward dependency" in doc.lower() or "no dependency" in doc.lower(),
            "Helpers docstring must state it has no dependency on other Living Loop modules",
        )

    def test_helpers_docstring_mentions_depth_layers(self):
        doc = self._get_docstring("hecsn.service.living_loop_helpers")
        # Should reference the dependency direction chain
        self.assertIn("Helpers", doc)
        self.assertIn("Records", doc)

    def test_records_docstring_mentions_layer(self):
        doc = self._get_docstring("hecsn.service.living_loop_records")
        self.assertTrue(
            "Layer A" in doc,
            "Records docstring must identify its depth layer (Layer A)",
        )

    def test_records_docstring_mentions_dependencies(self):
        doc = self._get_docstring("hecsn.service.living_loop_records")
        self.assertIn("helpers", doc.lower())

    def test_records_docstring_states_no_upward_imports(self):
        doc = self._get_docstring("hecsn.service.living_loop_records")
        self.assertTrue(
            "never imports from" in doc.lower() or "never import" in doc.lower(),
            "Records docstring must state it never imports from higher layers",
        )

    def test_policy_docstring_mentions_layer(self):
        doc = self._get_docstring("hecsn.service.living_loop_policy")
        self.assertTrue(
            "Layer B" in doc,
            "Policy docstring must identify its depth layer (Layer B)",
        )

    def test_policy_docstring_mentions_dependencies(self):
        doc = self._get_docstring("hecsn.service.living_loop_policy")
        self.assertIn("Records", doc)
        self.assertIn("Helpers", doc)

    def test_policy_docstring_states_no_upward_imports(self):
        doc = self._get_docstring("hecsn.service.living_loop_policy")
        self.assertTrue(
            "never imports from" in doc.lower(),
            "Policy docstring must state it never imports from higher layers",
        )

    def test_replay_docstring_mentions_layer(self):
        doc = self._get_docstring("hecsn.service.living_loop_replay")
        self.assertTrue(
            "Layer C" in doc,
            "Replay docstring must identify its depth layer (Layer C)",
        )

    def test_replay_docstring_mentions_dependencies(self):
        doc = self._get_docstring("hecsn.service.living_loop_replay")
        self.assertIn("Policy", doc)
        self.assertIn("Records", doc)

    def test_replay_docstring_states_no_upward_imports(self):
        doc = self._get_docstring("hecsn.service.living_loop_replay")
        self.assertTrue(
            "never imports from" in doc.lower(),
            "Replay docstring must state it never imports from higher layers",
        )

    def test_self_model_docstring_mentions_layer(self):
        doc = self._get_docstring("hecsn.service.living_loop_self_model")
        self.assertTrue(
            "Layer D" in doc,
            "Self-Model docstring must identify its depth layer (Layer D)",
        )

    def test_self_model_docstring_mentions_dependencies(self):
        doc = self._get_docstring("hecsn.service.living_loop_self_model")
        self.assertIn("Replay", doc)
        self.assertIn("Policy", doc)
        self.assertIn("Records", doc)

    def test_self_model_docstring_states_no_upward_imports(self):
        doc = self._get_docstring("hecsn.service.living_loop_self_model")
        self.assertTrue(
            "never imports upward" in doc.lower() or "never imports from" in doc.lower(),
            "Self-Model docstring must state it never imports upward",
        )

    def test_all_modules_have_dependency_direction_chain(self):
        """Every module docstring must include the full dependency direction chain."""
        module_fqns = [
            "hecsn.service.living_loop_helpers",
            "hecsn.service.living_loop_records",
            "hecsn.service.living_loop_policy",
            "hecsn.service.living_loop_replay",
            "hecsn.service.living_loop_self_model",
        ]
        for mod_name in module_fqns:
            doc = self._get_docstring(mod_name)
            self.assertIn(
                "Helpers",
                doc,
                f"{mod_name} docstring must include 'Helpers' in dependency direction",
            )
            self.assertIn(
                "Records",
                doc,
                f"{mod_name} docstring must include 'Records' in dependency direction",
            )


if __name__ == "__main__":
    unittest.main()
