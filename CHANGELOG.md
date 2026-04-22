# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-04-22

### Changed

- **BREAKING: `RevivifAI/derestriction` split taxonomy replaced.** The old
  `harmful` / `harmless` / `preservation` splits are gone. The dataset now
  ships three splits with a tri-valued policy semantics:
  - `restrict` — genuinely harmful prompts the model should continue to
    refuse (child harm, targeted violence, WMD, graphic sexual content,
    self-harm encouragement).
  - `derestrict` — refusal-target prompts the abliteration run should
    unlock (AdvBench / JailbreakBench misuse behaviors minus the
    restricted subset, Chinese Sensitive Topics).
  - `allow` — benign instructions and capability-preservation prompts
    (Alpaca, MMLU, GSM8K, ARC, MATH-500, HumanEval, JailbreakBench benign).
  The schema gains a `Restriction_Category` column populated on `restrict`
  rows identifying which filter category fired.
- **BREAKING: `HarmFilter.filter` now returns a `FilterResult`** containing
  both `kept` and `restricted` lists. Restricted rows are annotated with
  the firing category and routed into the `restrict` split instead of
  being silently dropped.
- `derestrictor.data.loader.VALID_SPLITS` is now
  `("restrict", "derestrict", "allow")`. All internal call sites
  (`abliterate`, `null_space`, `kl_monitor`, `eval.*`, `cli.main`,
  `scripts.*`) updated to request the new split names. The prior split
  names will not load.
- Build pipeline copies the **entire** upstream of every source — the
  per-split row caps (`TARGET_PER_SPLIT = 10_000`) and the preservation
  budgets are gone. Dedup priority is `restrict → derestrict → allow`,
  so ambiguous prompts land in the more conservative bucket.
- Parquet files are now written directly via `pyarrow.parquet.write_table`
  with a fixed three-column schema (Snappy compression), so HuggingFace
  does not auto-convert on upload.

### Added

- `HarmFilter` gains three new categories alongside the existing `child`
  and `sexual`:
  - `violence_extreme` — targeted assassination, murder, torture,
    kidnap-and-harm, terrorist planning against specific people or venues.
  - `weapons_mass_destruction` — bio / chem / nuke / radiological weapons,
    IEDs, pipe bombs, nerve agents, enriched fissile material.
  - `self_harm` — suicide encouragement and instruction, self-injury,
    pro-ana / pro-mia content.
- `build_dataset` runs the full harm filter against the `allow` pool too
  and writes any hits to `build/allow_split_safety_flags.jsonl` (in
  addition to moving them into `restrict`), so advertised-safe sources
  that slip unsafe content are auditable.
- `allenai/wildguardmix` (both `wildguardtrain` and `wildguardtest`) is
  now pulled as a gated source when `HF_TOKEN` is set and AI2 Responsible
  Use is accepted. Prompts are routed by the per-prompt `subcategory`
  field: `violence_and_physical_harm`, `sexual_content`,
  `mental_health_over-reliance_crisis`, and
  `causing_material_harm_by_disseminating_misinformation` go directly to
  `restrict` with `Restriction_Category = wildguardmix:<subcategory>`;
  every other subcategory (including `benign`, `cyberattack`,
  `fraud_assisting_illegal_activities`, `copyright_violations`,
  `private_information_individual`,
  `sensitive_information_organization_government`,
  `toxic_language_hate_speech`,
  `social_stereotypes_and_unfair_discrimination`,
  `disseminating_false_or_misleading_information_encouraging_disinformation_campaigns`,
  `defamation_encouraging_unethical_or_unsafe_actions`, and `others`)
  lands in `derestrict`, where the harm filter remains the authoritative
  second pass. If the token is missing or unauthorized the build logs a
  warning and continues on the open sources only. The summary JSON and
  dataset card record how many WildGuardMix prompts landed in each
  split.

### Removed

- **Proxy / derivative upstream sources have been dropped in favor of the
  originals.** The following sources are no longer pulled:
  - `LLM-LAT/harmful-dataset`, `Undi95/orthogonal-activation-steering-TOXIC`,
    `mlabonne/harmful_behaviors`, `wangzhang/abliterix-datasets` — all
    repackagings of AdvBench. Replaced by canonical
    [`walledai/AdvBench`](https://huggingface.co/datasets/walledai/AdvBench).
  - `mlabonne/harmless_alpaca`, `yahma/alpaca-cleaned` — forks of Alpaca.
    Replaced by the canonical
    [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)
    alone.
- Dropped `TARGET_PER_SPLIT`, `PRESERVATION_BUDGETS`, and the balanced
  per-source sampler in `build_dataset`. The full upstream is kept.

### Added (sources)

- [`JailbreakBench/JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
  (Chao et al., NeurIPS 2024) contributes 100 curated misuse behaviors
  (`derestrict`) and 100 curated benign parallels (`allow`) — an original
  research contribution, not an AdvBench fork.

## [1.2.0] - 2026-04-22

### Added

- Chinese Sensitive Topics prompts now feed the `harmful` split of the
  `RevivifAI/derestriction` dataset. Two new sources are pulled from
  [`MultiverseComputingCAI/llm-refusal-evaluation`](https://huggingface.co/datasets/MultiverseComputingCAI/llm-refusal-evaluation):
  the `ccp_sensitive` split (~1.36k rows) and the `deccp_censored` split
  (~95 rows). These are non-harmful but routinely-refused prompts useful
  for computing refusal directions on China-aligned instruct models.
- `HarmFilter` gained a second filter category — "sexual" — that drops
  graphic-sexual / pornographic / erotic prompts from the harmful pool
  via the same keyword + semantic-similarity pipeline already used for
  child-adjacent prompts. `FilterDrop` now carries a `category` field and
  the audit JSONL records it alongside the stage and trigger.

### Changed

- Dataset card (`Safety notice` + `harmful` section) regenerated by
  `derestrictor.scripts.upload_dataset` now documents the sexual-content
  filter and the new Chinese-sensitive-topic sources.

### Removed

- Dropped the stale `llm-derestrictor/preservation.txt` budget entry in
  `PRESERVATION_BUDGETS` (and all rows labeled with that source). The
  curated prompt file was removed in 1.1.0 but the budget entry and the
  old parquet it populated lingered; they are gone for good now.

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
