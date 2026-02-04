# Available commands for Sklearn-Wrap

# Show all available commands
default:
    @just --list

# Install dependencies and pre-commit
install:
    uv sync --group dev
    uvx pre-commit install

# Run tests
test:
    uv run pytest -v

# Run fast tests (excludes slow and integration tests)
test-fast:
    uv run pytest -m "not slow and not integration" -v

# Run slow tests (includes integration tests)
test-slow:
    uv run pytest -m "slow or integration" -v

# Run tests with coverage
test-cov:
    uv run pytest --cov=sklearn_wrap --cov-report=html --cov-report=term

# Run docstring examples
doctest:
    uv run pytest --doctest-modules --doctest-continue-on-failure src/sklearn_wrap

# Run marimo example notebook interactively
example file='':
    uv run marimo edit examples/{{file}}

# Test all example notebooks
test-examples:
    uv run pytest tests/test_examples.py -m example -n auto -v

# Run linters and type checkers
lint:
    uv run ruff check src tests
    uv run ty check src

# Format and fix code (via pre-commit)
fix:
    uvx pre-commit run --all-files --show-diff-on-failure

# Build documentation
build:
    uv run mkdocs build --clean

# Serve documentation locally
serve: build
    @echo "###### Starting local server. Press Control+C to stop server ######"
    uv run mkdocs serve -a localhost:8080

# Clean build artifacts
clean:
    rm -rf .nox
    rm -rf build dist *.egg-info
    rm -rf .pytest_cache .ty_cache .ruff_cache
    rm -rf htmlcov .coverage coverage.xml
    rm -rf site
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

# Run all quality checks (fix, test)
all: fix test
