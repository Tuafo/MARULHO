from __future__ import annotations

from typing import Any


class ManagerBoundModule:
    """Forward attribute access to the owning manager when bound.

    Seam objects use this so extracted helpers can run either as standalone
    modules in direct tests or as manager-bound delegates in production.
    """

    def __init__(self, manager: Any | None = None) -> None:
        object.__setattr__(self, "_manager", manager)

    def __getattr__(self, name: str) -> Any:
        manager = object.__getattribute__(self, "_manager")
        if manager is None:
            raise AttributeError(name)
        return getattr(manager, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_manager":
            object.__setattr__(self, name, value)
            return
        try:
            manager = object.__getattribute__(self, "_manager")
        except AttributeError:
            object.__setattr__(self, name, value)
            return
        if manager is None or manager is self:
            object.__setattr__(self, name, value)
            return
        setattr(manager, name, value)
