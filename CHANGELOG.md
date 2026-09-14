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

### Documentation

- Expand Quick Start with executable price, return, risk and portfolio examples, including expected results.

- Rewrite the decorators reference with concise descriptions and runnable examples; hide its private conversion helper.

- Add a generated Markdown copy of the guides and API under `docs/reference`, with a synchronization check for pull requests.
- Publish GitHub Pages on changes to `master` and manual runs.



### Added

- Portfolio P&L, interval returns adjusted for external flows and time-weighted
  returns with explicit beginning/end flow timing.
- Covariance-based portfolio variance, volatility and per-asset risk
  contributions, plus historical Expected Shortfall and compounded-wealth CDaR.
- Portfolio valuation, gross/net exposure, leverage, weights, concentration,
  weighted returns and executed-trade turnover in `quantkit.portfolio`, with
  strict NumPy/pandas shape and label alignment and explicit cash handling.
- Drawup family, the mirror image of drawdown (Vecer, 2006):
  `expanding.drawup` and `stats.max_drawup`.
- Maximum drawdown details: `max_drawdown_peak`, `max_drawdown_valley`,
  `max_drawdown_recovery`, `max_drawdown_duration`,
  `max_drawdown_recovery_duration`, `longest_drawdown_duration` and
  `average_drawdown`.
- Return and drawdown ratios: `annualized_return`, `calmar_ratio` and
  `sterling_ratio`.
- Downside risk: `downside_deviation`, `upside_deviation`, `kappa`,
  `omega_ratio` and `sortino_ratio`.
- Gain and loss statistics: `average_gain`, `average_loss`,
  `gain_loss_ratio`, `up_period_percent` and `down_period_percent`.
- Historical `value_at_risk`.
- Benchmark-relative statistics, aligned by index through `utils.align`:
  `beta`, `alpha`, `correlation`, `r_squared`, `bull_beta`, `bear_beta`,
  `treynor_ratio`, `tracking_error`, `information_ratio`, `up_capture`,
  `down_capture`, `overall_capture` and `batting_average`.

  Definitions follow Morningstar's Custom Calculation Data Points
  (October 2016). Undefined results (zero denominators, no valid
  observations) are NaN, never inf.

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
- Empty input raised instead of reducing to NaN. `decorators.reduce_array_wrap`
  built the reduced container from the first row, so any reducer given a
  2-dimensional input with zero rows raised `IndexError`; it now builds the
  container from the columns. `stats.total_returns`, `volatility`, `drawdown`,
  `max_drawdown` and `sharpe_ratio` also raised or warned on empty input and
  now return NaN, one value per column, like the rest of the reducers.
- `stats.sharpe_ratio` reduced 2-dimensional input along the wrong axis, so
  it returned one value per row instead of one per column, divided by a
  single deviation computed over the whole array, and gave back a bare
  ndarray for a `DataFrame`. It now reduces per column through
  `reduce_array_wrap`, subtracts an array-like `risk_free` row by row
  (aligned by index for pandas) and returns NaN, never inf, when the excess
  return has no deviation.
- `stats.total_returns` returned a bare ndarray for a `DataFrame` instead of
  a `Series` indexed by the columns, raised on integer prices with
  `relative=True` and on a 1-dimensional input without a valid observation.
  It now goes through `reduce_array_wrap` like every other reducer and gives
  NaN in both cases.
- `stats.drawdown` measured every column of a 2-dimensional input against
  the maximum of the whole array instead of the maximum of its own column,
  so the result for a `DataFrame` did not match the result for each of its
  columns. It also raised on a column without a valid observation and
  returned a 2-dimensional array for an all-NaN 1-dimensional input; both
  now give NaN.

[Unreleased]: https://github.com/asdf8601/quantkit/compare/v0.0.0...HEAD
