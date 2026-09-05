# CHANGELOG

<!--

Guiding Principles 

    Changelogs are for humans, not machines.
    There should be an entry for every single version.
    The same types of changes should be grouped.
    Versions and sections should be linkable.
    The latest version comes first.
    The release date of each version is displayed.
    Mention whether you follow Semantic Versioning.

Types of changes

    `Added` for new features.
    `Changed` for changes in existing functionality.
    `Deprecated` for soon-to-be removed features.
    `Removed` for now removed features.
    `Fixed` for any bug fixes.
    `Security` in case of vulnerabilities.

All notable changes to this project will be documented in this file.

-->

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


### Added

- None

### Changed

- Build with `uv_build` instead of setuptools. The version is now static in
  `pyproject.toml` (`0.1.0`), versioneer and `_version.py` are gone, and
  `quantkit.__version__` is read from the installed package metadata.
- Dependencies live in PEP 735 dependency groups (`test`, `lint`, `docs`,
  `dev`) with a committed `uv.lock`; the `requirements*.txt` files are
  removed. CI runs on `uv`.
- Lint and format with ruff (replacing flake8, black and pydocstyle) and
  type-check with ty.
- Python 3.10 or newer is required.
- Replace the removed pandas `__array_wrap__` with `quantkit.utils.array_wrap`,
  adding support for pandas 2 and 3. The temporary `numpy<2, pandas<2` pins
  are gone.

### Fixed

- `conventions.ArrayLike` referenced `np.array` (a function) instead of
  `np.ndarray`.
- `stats.sharpe_ratio` annotated `risk_free` as `float` while the docstring
  and tests use array-like values; the annotation now allows both.

[Unreleased]: https://github.com/asdf8601/quantkit/compare/v0.0.0...HEAD
