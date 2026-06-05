"""MARULHO Stage-0 reference implementation.

This package provides a minimal executable path for the architecture described
in marulho-16.md, focused on emergent unsupervised concept formation.
"""

from .config.model_config import MarulhoConfig

__all__ = ["MarulhoConfig"]
