
##@ Utility
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

dev:
	python3 -m venv .venv
	./.venv/bin/pip install -e .[dev]


check:
	ruff check src tests
	mypy src tests

format:
	ruff format src tests

test:
	pytest -v
