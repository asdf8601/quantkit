"""QuantKit init file."""

from importlib.metadata import version

from . import core, decorators, expanding, portfolio, rolling, stats, utils

__version__ = version("quantkit")


__all__ = [
    "core",
    "decorators",
    "stats",
    "utils",
    "expanding",
    "rolling",
    "portfolio",
]
