"""Compatibility import for the fused route/vote evaluation benchmarks."""

from marulho.core.fused_route_vote_cuda import (  # noqa: F401
    fused_route_vote_available,
    fused_route_vote_cuda,
)

__all__ = [
    "fused_route_vote_available",
    "fused_route_vote_cuda",
]
