# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-04-22

### Changed

- **Project reorganized into a modern src-layout.** All code now lives under a
  single importable package, `derestrictor`, inside `src/derestrictor/`. The
  former flat `src/` and `utils/` top-level packages are gone.
- Distribution name renamed from `abliteration` to `derestrictor`
  (`pyproject.toml`'s `[project].name`). The repository URL and `git clone`
  path are unchanged.
- Package submodules grouped by concern:
  - `derestrictor.core` — algorithms (`abliterate`, `null_space`,
    `feature_surgery`, `kl_monitor`)
  - `derestrictor.cli` — Click + questionary UI (`main`, `components`,
    `feature_surgery`)
  - `derestrictor.config` — persistent settings (`manager`)
  - `derestrictor.data` — dataset loading and harm filtering (`loader`,
    `harm_filter`)
  - `derestrictor.models` — model loading utilities (`utils`)
  - `derestrictor.eval` — refusal detection and scoring (`detector`,
    `scanner`, `calibration`, `compare`, `harness`)
  - `derestrictor.export` — GGUF export (`gguf`)
  - `derestrictor.sae` — sparse-autoencoder helpers (`loader`)
  - `derestrictor.scripts` — user-runnable pipelines
    (`abliteration_search`, `batch_quantize`, `build_dataset`,
    `upload_dataset`, `convert_to_bf16`)
- All `from src.X` / `from utils.Y` imports rewritten to the new
  `derestrictor.<subpackage>.<module>` paths, including lazy in-function
  imports.

### Added

- `tests/` directory with pytest scaffolding (`conftest.py`, `test_smoke.py`).
- `docs/` directory:
  - `docs/assets/cli_example.jpg` (moved from the repo root).
  - `docs/papers/` — archived copies of every research reference cited in the
    README, so the repo is usable offline and immune to link rot. Includes
    the Arditi (2024), Zou (2023), and Fang (2025) arXiv PDFs, plus a
    markdown snapshot of Jim Lai's Norm-Preserving Biprojected Abliteration
    Hugging Face blog post.
- New `derestrictor-*` console entry points:
  - `derestrictor` (interactive CLI), `derestrictor-feature-surgery`,
    `derestrictor-search`, `derestrictor-quantize`,
    `derestrictor-build-dataset`, `derestrictor-upload-dataset`,
    `derestrictor-convert-bf16`.
- `python -m derestrictor` entry point via `derestrictor.__main__`.
- `[tool.pytest.ini_options]` section in `pyproject.toml` with a
  `requires_cuda` marker.
- Top-level `CHANGELOG.md` (this file), `CONTRIBUTING.md`, and `.env.example`.

### Deprecated

- `abliterate` and `feature-surgery` console entry points remain as
  backward-compatible aliases for `derestrictor` and
  `derestrictor-feature-surgery` respectively. They will be removed in 2.0.0.

### Infrastructure

- `.pre-commit-config.yaml`: raised `check-added-large-files --maxkb` from
  1024 to 20480 so archived research PDFs can be committed without hitting
  the hook. Non-archival large files should still be avoided.

## [1.0.0] - 2026-04-22

### Added

- Initial public release of the Abliteration Toolkit (pre-reorg layout).
