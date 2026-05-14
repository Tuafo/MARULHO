"""Autonomy planning helpers for Terminus.

This module owns gap-based focus planning, candidate-bank shaping, shortlist
sizing, provider curriculum prioritization, and query-family scoring. It is
the explicit seam for autonomous acquisition decisions.
"""

from __future__ import annotations

from typing import Any

from hecsn.service.terminus_autonomy import TerminusAutonomyMixin as _TerminusAutonomyMixin


class AutonomyPlanner(_TerminusAutonomyMixin):
    """Autonomy planner with explicit dependency access."""

    def __init__(self, dependencies: Any | None = None) -> None:
        object.__setattr__(self, "_dependencies", dependencies)

    @property
    def dependencies(self) -> Any:
        return object.__getattribute__(self, "_dependencies")


def _install_dependency_forwarders(cls: type, names: tuple[str, ...]) -> None:
    for raw_name in names:
        name = str(raw_name)
        if not name or hasattr(cls, name):
            continue

        def _get(self: AutonomyPlanner, *, _name: str = name) -> Any:
            dependencies = object.__getattribute__(self, "_dependencies")
            if dependencies is None:
                raise AttributeError(_name)
            return getattr(dependencies, _name)

        def _set(self: AutonomyPlanner, value: Any, *, _name: str = name) -> None:
            dependencies = object.__getattribute__(self, "_dependencies")
            if dependencies is None:
                object.__setattr__(self, _name, value)
                return
            setattr(dependencies, _name, value)

        setattr(cls, name, property(_get, _set))


_install_dependency_forwarders(AutonomyPlanner, (
    "_brain_config",
    "_concept_store",
    "_geometric_curiosity",
    "_interaction_pipeline",
    "_normalize_provider_curriculum",
    "_selected_evidence_weight_map",
    "_source_text_overlap",
))


AutonomyPlannerMixin = AutonomyPlanner
