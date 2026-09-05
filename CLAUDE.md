# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`quantkit` is a small finance-statistics library (returns, drawdown, volatility, Sharpe) built on numpy and pandas only. Source lives under `src/quantkit/` (src layout). Status is alpha and the README says "Very WIP".

## Commands

Everything runs through `uv`. `uv sync` creates `.venv` with the package (editable) plus the `dev` dependency group, which pulls in `test`, `lint` and `docs`. CI uses Python 3.10, the minimum supported version.

```bash
uv sync                                       # full dev environment
uv sync --no-default-groups --group test      # what CI installs for tests
```

Tests (CI runs the first form):

```bash
uv run pytest tests -v --cov                  # full suite with coverage
uv run pytest tests/test_stats.py             # one file
uv run pytest tests/test_stats.py::test_total_returns_relative
uv run pytest -k drawdown                     # by keyword
uv run pytest -n auto                         # parallel, pytest-xdist is installed
uv run pytest -W error::pytest.PytestRemovedIn10Warning   # catch pytest 10 breakage
```

Lint, format and types (CI runs all three, bare, from the repo root):

```bash
uv run ruff check                             # add --fix to apply
uv run ruff format --check                    # drop --check to reformat
uv run ty check
```

Ruff config is in `pyproject.toml`: line length 79, numpy docstring convention, rules E/F/W/I/D. `tests/` and `docs/` are excluded from ruff, matching the old flake8 scope, but ty checks the whole tree including tests.

Docs (Sphinx + autoapi over `src/`; published to GitHub Pages on release):

```bash
uv sync --group docs
uv run make -C docs html        # output in docs/build/html
```

Build: `uv build` (backend is `uv_build`, pure Python only). The version is the static `project.version` in `pyproject.toml`; bump it with `uv version --bump patch|minor|major` and tag the commit `vX.Y.Z` to release. Pushing a `v*` tag runs build, test, lint, then publishes docs to Pages and uploads to PyPI with `uv publish`. Record changes under `[Unreleased]` in `CHANGELOG.md` (Keep a Changelog format).

`uv.lock` is committed. Run `uv lock` after touching dependencies; CI uses `--locked` and fails if the lockfile is stale.

## Architecture

The library has one central contract: **every public function accepts a numpy array or a pandas Series/DataFrame, and returns the same container type as its input**, preserving index, columns and name for pandas. The modules are organised by the shape of that transformation, not by financial topic.

| Module | Shape | Contents |
|---|---|---|
| `core` | series -> series | `returns`, `cum_returns`, `rebase` |
| `expanding` | series -> series, expanding window | `drawdown` |
| `rolling` | series -> series, rolling window | `volatility` (wraps `stats.volatility` through pandas `.rolling().apply`) |
| `stats` | series -> one number per column | `total_returns`, `volatility`, `drawdown`, `max_drawdown`, `sharpe_ratio` |
| `decorators` | plumbing | `numpy2pandas_args_wrapper`, `array_output_wrapper`, `reduce_array_wrap` |
| `utils` | plumbing | `array_wrap`, `iloc`, `first_valid_index`, `last_valid_index` |
| `conventions` | constants | `BYEAR = 261` (annualisation default), `ArrayLike` alias |

How the contract is implemented, and the pattern to follow when adding a function:

1. Get the raw buffer with `x.__array__()` so the same numpy code runs for both container types.
2. Compute with numpy ufuncs writing into an `out` buffer (`out=` kwarg), allocated with `np.zeros_like(x, float)` if the caller did not pass one. Leading positions that cannot be computed are set to `np.nan`.
3. Rewrap with `utils.array_wrap(like, values)`. This is the project's replacement for pandas' removed `__array_wrap__` and is why pandas 2 and 3 work; it rebuilds the Series/DataFrame with `copy=False` over the original index and columns.
4. Reducers (the `stats` module) instead call `decorators.reduce_array_wrap(obj, res)`, which turns a per-column result into a Series indexed by the input columns for 2D input and leaves a scalar for 1D input.

Functions that need pandas semantics on numpy input (rolling windows) use `@numpy2pandas_args_wrapper(0)` to coerce the argument first and `@array_output_wrapper(0)` to convert the result back. `expanding.drawdown` uses only the first decorator, so a numpy input comes back as a Series.

NaN handling is deliberate throughout: reducers use nan-aware numpy (`nanstd`, `nanmin`, `nancumprod`) and locate the usable range with `first_valid_index`/`last_valid_index`, which work per column on 2D input.

Known rough edges worth knowing before touching them:

- `rolling` is not imported in `quantkit/__init__.py`, so `qnt.rolling` is not available; use `from quantkit import rolling`.
- `core.rebase` skips the `array_wrap` step and always returns an ndarray, unlike its neighbours.
- `decorators.array_output_wrapper` only handles a single return value (noted in a TODO in the source).

## Tests

Tests live in `tests/` and are plain pytest functions, usually one numpy and one pandas variant per behaviour, compared with `np.testing.assert_almost_equal` or `pd.testing.assert_*_equal`. `tests/utils.py::std2series` uses sympy to construct a three-element series with an exact target standard deviation, which is why sympy is a test dependency. Parametrize argvalues must be concrete collections (lists), not generators or `zip` objects: pytest 10 rejects one-shot iterables.
