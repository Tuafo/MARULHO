from .manager import HECSNServiceManager


def create_app(*args, **kwargs):
    from .api import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app", "HECSNServiceManager"]
