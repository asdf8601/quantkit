"""QuantKit init file."""

from importlib.metadata import version

from . import core, decorators, expanding, portfolio, stats, utils

__version__ = version("quantkit")


__all__ = ["core", "decorators", "stats", "utils", "expanding", "portfolio"]
