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
