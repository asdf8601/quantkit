# quantkit

Very WIP! Finance functions.

## Installation

```bash
uv add git+https://github.com/asdf8601/quantkit
# or, outside a uv project
uv pip install git+https://github.com/asdf8601/quantkit
```

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
