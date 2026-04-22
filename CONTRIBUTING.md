# Contributing

Thanks for your interest in contributing to `derestrictor`. This document
covers the local development setup and the checks your changes are expected
to pass before review.

## Development setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management
and a pinned Python version (see `.python-version`, currently 3.12).

```bash
git clone https://github.com/RevivifAI/llm-derestrictor
cd llm-derestrictor

# Install runtime + dev deps into .venv/
uv sync --group dev

# One-time: enable local git hooks
uv run pre-commit install
```

If your work needs HuggingFace access (model download, dataset upload), copy
`.env.example` to `.env` and fill in `HF_TOKEN`. The `.env` file is
gitignored.

## Project layout

All library code lives under `src/derestrictor/`, organized into cohesive
subpackages. See the `## Changed` entry in
[`CHANGELOG.md`](CHANGELOG.md) under 1.1.0 for the full mapping. New modules
should land in the subpackage that matches their concern; do not add new
top-level packages.

User-runnable pipeline scripts live in `src/derestrictor/scripts/` and are
exposed as `derestrictor-*` console entry points via
`[project.scripts]` in `pyproject.toml`.

## Checks

Before opening a PR, please make sure the following commands succeed from the
repo root:

```bash
uv run pre-commit run --all-files   # formatters + ruff + misc hooks
uv run pytest                        # unit tests
uv run ruff check src tests          # standalone linter pass
uv run ruff format --check src tests # formatter in check mode
```

`pre-commit` runs `ruff check --fix` and `ruff format` automatically on
commit; fixing issues up-front keeps commit messages clean.

## Coding standards

- Python 3.12-compatible code. Target-version is pinned in `pyproject.toml`.
- Ruff governs style and lints — see `[tool.ruff]` in `pyproject.toml` for the
  enabled rule set (pycodestyle, flake8-bugbear, flake8-simplify, isort,
  pydocstyle with Google convention, bandit, etc.).
- Top-level imports only for ordinary dependencies. Lazy, in-function imports
  are reserved for optional heavy ML deps (torch extensions, SAE tooling) or
  breaking known circular imports — document the reason in a short comment.
- Docstrings follow Google style. Public APIs carry type hints.
- Relative imports are banned (`flake8-tidy-imports`). Use absolute imports:
  `from derestrictor.core.abliterate import ...`

## Tests

- Tests live in `tests/`. Pytest config is in `pyproject.toml` under
  `[tool.pytest.ini_options]`.
- Mark GPU-only tests with `@pytest.mark.requires_cuda`; the marker's purpose
  is to let CPU-only CI skip them explicitly.
- Prefer small, deterministic pure-math tests over loading real HF models.

## Commit messages

Write focused, self-contained commits. Explain the *why*, not just the
*what*, in the body. The project does not currently enforce a strict
conventional-commits format, but a short imperative subject line is
appreciated.
