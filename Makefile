.PHONY: help install lint format typecheck test cov figures clean

help: ## Show this help message.
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## / {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package in editable mode with dev extras.
	python -m pip install -e ".[dev]"
	pre-commit install

lint: ## Run ruff linter.
	ruff check src/ tests/ scripts/

format: ## Apply ruff formatting.
	ruff format src/ tests/ scripts/

typecheck: ## Run mypy in strict mode on the package.
	mypy src/btflow

test: ## Run the test suite.
	pytest

cov: ## Run tests with coverage report.
	pytest --cov=btflow --cov-report=term-missing

figures: ## Regenerate synthetic sample figures under assets/synthetic/.
	MPLBACKEND=Agg python scripts/regenerate_assets.py

clean: ## Remove build, cache, and coverage artefacts.
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
