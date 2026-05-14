from __future__ import annotations

from typing import Any, Iterable


class ExplicitOwnerModule:
    """Base for transition modules that keep an explicit owner reference.

    This intentionally does not implement catch-all ``__getattr__`` or
    ``__setattr__``. Any compatibility access back to the owner must be
    installed as a named property via ``install_owner_forwarders``.
    """

    def __init__(self, manager: Any | None = None) -> None:
        object.__setattr__(self, "_manager", manager)

    def _bound_module(self, attribute_name: str) -> Any:
        module = getattr(self, attribute_name, None)
        if module is not None:
            return module
        manager = object.__getattribute__(self, "_manager")
        if manager is not None:
            module = getattr(manager, attribute_name, None)
            if module is not None:
                return module
            return manager
        return self


def install_owner_forwarders(cls: type, names: Iterable[str]) -> None:
    """Install explicit owner-backed properties for transition dependencies."""

    for raw_name in names:
        name = str(raw_name)
        if not name or hasattr(cls, name):
            continue

        def _get(self: ExplicitOwnerModule, *, _name: str = name) -> Any:
            manager = object.__getattribute__(self, "_manager")
            if manager is None:
                raise AttributeError(_name)
            return getattr(manager, _name)

        def _set(self: ExplicitOwnerModule, value: Any, *, _name: str = name) -> None:
            manager = object.__getattribute__(self, "_manager")
            if manager is None:
                object.__setattr__(self, _name, value)
                return
            setattr(manager, _name, value)

        setattr(cls, name, property(_get, _set))
