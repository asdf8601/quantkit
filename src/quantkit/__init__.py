"""QuantKit init file."""

from importlib.metadata import version

from . import core, decorators, expanding, stats, utils

__version__ = version("quantkit")


__all__ = ["core", "decorators", "stats", "utils", "expanding"]
