# quantkit

Very WIP! Finance functions.

## Documentation

- [Read the guides and complete API on GitHub](docs/README.md)
- [Browse the documentation website](https://asdf8601.github.io/quantkit/)

## Installation

```bash
uv add git+https://github.com/asdf8601/quantkit
# or, outside a uv project
uv pip install git+https://github.com/asdf8601/quantkit
```

### Interactive tour

Three marimo notebooks teach Quantkit through English explanations, visible
Python cells, real historical prices and hypothetical portfolio positions:

The examples use the recommended alias `import quantkit as qnt`, with calls
such as `qnt.core.rebase(prices)` and
`qnt.portfolio.valuation.net_asset_value(position_values, cash=cash)`.

| Notebook | Contents |
| --- | --- |
| [Quickstart](examples/notebooks/01_quickstart.py) | Returns, rebasing to 100, annualized metrics and drawdowns |
| [Advanced](examples/notebooks/02_advanced.py) | Benchmark comparisons, rolling volatility and historical tail risk |
| [Portfolio](examples/notebooks/03_portfolio.py) | Invented holdings, valuation, exposures, portfolio returns, risk contributions and cash flows |

From the repository root, open any notebook in the editor to read, change and
run the code alongside its results:

```bash
uv run --group examples marimo edit examples/notebooks/01_quickstart.py
```

The calculation cells also display their source in the app view
(`marimo run`) and exported HTML, so each result can be traced to its code.
All three read the same local CSV; changing controls never downloads data.
Marimo, Plotly and yfinance are optional example dependencies, separate from
the library's runtime requirements.

Validate and export a notebook with its default selections:

```bash
uv run --group examples marimo check examples/notebooks/01_quickstart.py
uv run --group examples marimo export html examples/notebooks/01_quickstart.py -o /tmp/quantkit-quickstart.html
uv run --group examples pytest tests/test_notebooks.py -q
```

The HTML captures the code and current results. Open the notebook with
`marimo edit` to experiment with the implementation and recalculate results.

### Example data

Initialize the shared historical dataset from the repository root:

```bash
uv run --group examples python examples/init_data.py
```

The script downloads daily SPY, QQQ, TLT and GLD data for 2018–2025 from
Yahoo Finance once. It saves `examples/data/market/prices.csv` and a
`metadata.json` with the source, download date, settings and SHA-256 checksum.
These files belong in version control so every example uses the same snapshot.
Subsequent runs verify and reuse the files without accessing Yahoo. An
incomplete or modified snapshot raises an error instead of being overwritten.

Read the snapshot offline with pandas:

```python
from pathlib import Path
import pandas as pd

data_path = Path("examples/data/market/prices.csv")  # from the repo root
data = pd.read_csv(data_path, parse_dates=["date"])
prices = data.pivot(index="date", columns="ticker", values="adjusted_close")
```

`close` is Yahoo's Close field with automatic adjustment disabled;
`adjusted_close` is its Adj Close field for performance analysis. Dividend
amounts and split ratios are retained separately. Prices are in USD; dates
are daily market session labels. No missing prices are filled. Portfolio
positions and cash flows in the tour are hypothetical.

The download parameters follow the
[yfinance API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
Downloader dependencies are isolated in the optional `examples` group.

### Developers

```bash
git clone https://github.com/asdf8601/quantkit
cd quantkit
uv sync                      # .venv with the package and every dependency group
uv run pytest
uv run ruff check
uv run ruff format --check
uv run ty check
```
