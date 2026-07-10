"""Thin HTTP/UI adapter over MarulhoBrain."""

from .brain_manager import MarulhoBrainServiceManager


def create_app(*args, **kwargs):
    from .api import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["MarulhoBrainServiceManager", "create_app"]
