"""QuantKit init file."""
from . import core, decorators, stats, utils, expanding
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


__all__ = ["core", "decorators", "stats", "utils", "expanding"]
