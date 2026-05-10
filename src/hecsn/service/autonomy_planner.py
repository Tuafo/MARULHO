"""Autonomy planning helpers for Terminus.

This module owns gap-based focus planning, candidate-bank shaping, shortlist
sizing, provider curriculum prioritization, and query-family scoring. It is
the explicit seam for autonomous acquisition decisions and can run either as a
standalone manager-bound module or as a compatibility alias for the legacy
terminus autonomy mixin.
"""

from __future__ import annotations

from typing import Any

from hecsn.service.manager_bound_module import ManagerBoundModule
from hecsn.service.terminus_autonomy import TerminusAutonomyMixin as _TerminusAutonomyMixin


class AutonomyPlanner(ManagerBoundModule, _TerminusAutonomyMixin):
    """Manager-bound autonomy planner with legacy mixin behavior."""

    def __init__(self, manager: Any | None = None) -> None:
        super().__init__(manager)


AutonomyPlannerMixin = AutonomyPlanner

