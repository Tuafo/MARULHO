"""Autonomy planning helpers for Terminus.

This module owns gap-based focus planning, candidate-bank shaping, shortlist
sizing, provider curriculum prioritization, and query-family scoring. It is
the explicit seam for autonomous acquisition decisions and can run either as a
standalone manager-bound module or as a compatibility alias for the legacy
terminus autonomy mixin.
"""

from __future__ import annotations

from hecsn.service.manager_bound_module import ExplicitOwnerModule, install_owner_forwarders
from hecsn.service.terminus_autonomy import TerminusAutonomyMixin as _TerminusAutonomyMixin


class AutonomyPlanner(ExplicitOwnerModule, _TerminusAutonomyMixin):
    """Manager-bound autonomy planner with legacy mixin behavior."""


install_owner_forwarders(AutonomyPlanner, (
    "_brain_config",
    "_concept_store",
    "_geometric_curiosity",
    "_interaction_pipeline",
    "_normalize_provider_curriculum",
    "_selected_evidence_weight_map",
    "_source_text_overlap",
))


AutonomyPlannerMixin = AutonomyPlanner
