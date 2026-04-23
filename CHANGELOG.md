# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-04-22

### Added

- New `derestrictor.core.sharded_ablate` module implementing one-shard-at-a-
  time streaming ablation. Reads each safetensors shard with
  `safetensors.safe_open`, applies the kernel on the configured device,
  and writes a modified output shard, never materializing the full
  state_dict. Memory cap is one shard plus per-tensor scratch, which is
  the only way frontier-scale (235B / 480B) MoE models fit on a consumer
  GPU. Mirrors the `jim-plus/llm-abliteration/sharded_ablate.py` read ->
  upcast -> modify -> downcast -> write loop and reuses the same kernel
  dispatch as the in-memory path.
- `AbliterationConfig.streaming` selector
  (`"auto"` / `"in_memory"` / `"sharded"`, default `"auto"`). Auto mode
  picks streaming when the on-disk model footprint exceeds 0.6x available
  GPU RAM (or 0.4x system RAM on CPU-only runs). `run_abliteration`
  releases the in-memory model and clears CUDA cache before invoking the
  streaming pipeline so the read-modify-write loop has the full GPU
  available.
- First-class support for fused 3-D MoE expert tensors. New
  `apply_kernel_to_expert_tensor` runs any 2-D ablation kernel across
  every expert in a fused MoE weight (`[E, O, I]` Mixtral or `[E, I, O]`
  Qwen2.5-MoE / GPT-OSS), with auto-detection of layout from shape and
  model family.  `AbliterationConfig.moe_fused_layout` (`"auto"` /
  `"eoi"` / `"eio"`) overrides the auto-detect when needed. Previously
  these tensors were silently skipped by the linear-layer walker, leaving
  MoE MLPs un-ablated.
- `is_moe_model` and `iter_expert_tensors` helpers in
  `derestrictor.models.utils` plus the `ExpertTensorInfo` dataclass.
  Detects MoE via `num_local_experts` / `num_experts` /
  `moe_num_experts` / `n_routed_experts` (descending into `text_config`
  for VL configs) or via a module-tree walk for `block_sparse_moe` /
  `mlp.experts` / `.experts.<digit>` markers.
- `get_layer_type_from_name` now recognizes the fused MoE parameter
  conventions (`block_sparse_moe.experts.w1/w2/w3`,
  `experts.gate_up_proj`, `experts.down_proj`,
  `mlp.experts.gate_up_proj`) so the kernel-dispatch and
  direction-role logic reuses the existing canonical layer types.
- `AbliterationConfig.measurement_quant`
  (`"none"` / `"4bit"` / `"8bit"`, default `"none"`) loads the model with
  bitsandbytes quantization for the activation-measurement phase only.
  Final ablation is always performed on the unquantized weights via the
  streaming pipeline (setting this implies sharded streaming). Matches
  the "measurement on a 4-bit quant, assembly on the bf16 model" recipe
  from the Norm-Preserving Biprojected Abliteration article comments.
- `select_ablation_kernel`, `dispatch_2d_kernel`,
  `resolve_ablation_for_tensor`, and `write_safetensors_index` helpers
  refactored out of `abliterate_model` so the in-memory and streaming
  paths share one kernel dispatch and one shard-index writer.
- New CLI flags `--streaming`, `--moe_fused_layout`, and
  `--measurement_quant`. The interactive wizard prompts for the
  streaming pipeline, auto-explains MoE layout when the architecture is
  detected, and offers measurement-only quantization. All three are
  persisted into the abliteration metadata block.

## [2.1.0] - 2026-04-22

### Added

- New `apply_householder_rotation` ablation kernel. Geodesic Rodrigues
  rotation transcribed from `jim-plus/llm-abliteration`, with the trig-free
  cos/sin derivation, antipodal special case, and post-op renorm clamp.
  Isometric by construction. Operating points: `scale_factor=0` is identity,
  `1.0` is full rotation `s -> t` (or full reflection at the antipode),
  `2.0` is reflection through the hyperplane orthogonal to `s`. The
  rank-1 form avoids materializing a `[D, D]` rotation matrix.
- New `apply_directional_scaling` ablation kernel. Rank-1 directional
  scaling `W + (alpha - 1) * (W * s) ⊗ s` with double-tap cancellation
  for exact projection and per-row renormalization. Negative
  `scale_factor` amplifies along `s` instead of attenuating, giving the
  upstream `--invert` / induction behavior for any kernel via the new
  `invert_ablation` flag.
- `AbliterationConfig.ablation_kernel` selector (`"per_neuron"`,
  `"frobenius"`, `"householder"`, `"directional"`). When set, this is
  authoritative; when `None` (the default), the legacy
  `use_per_neuron_norm` / `norm_preservation` boolean dispatch is
  preserved unchanged.
- `AbliterationConfig.invert_ablation` flag flips the kernel multiplier
  sign at the call boundary so `1.0` = removal, `0.0` = identity,
  `-1.0` = amplification, composing with all four kernels.
- `magnitude_sparsify` helper plus `AbliterationConfig.direction_sparsity`
  and `AbliterationConfig.per_layer_sparsity` config fields. The refusal
  direction is sparsified per layer after orthogonalization and before
  the kernel call (renormalized after), enabling Jim Lai's
  "sparsity 0.001 on layers 35-41" Gemma 3 pattern via an explicit
  per-layer dict override.
- `orthogonalize_against_harmless` and `compute_biprojected_direction`
  now accept a `two_pass` argument (default-on via the new
  `AbliterationConfig.two_pass_orthogonalization` field). The second
  pass kills the residual `r * h_hat` component left by float
  cancellation when harmful and harmless means have very high cosine
  similarity (e.g., Gemma 3). One extra dot product, strictly more
  stable.
- `token_position` now accepts `"first_generated"` and
  `"second_generated"`. Routes through `model.generate(...,
  return_dict_in_generate=True, output_hidden_states=True)` and slices
  the per-layer hidden state at the requested generated step, matching
  the upstream `welford_gpu_batched_multilayer_float32` measurement
  convention.
- `--ablation-kernel`, `--invert` / `--no-invert`,
  `--direction-sparsity`, `--two-pass-orthogonalization` /
  `--no-two-pass-orthogonalization`, and the expanded `--token-position`
  enum are wired through `derestrictor.core.abliterate.main` and the
  interactive wizard's settings dict; all new fields round-trip into
  `abliteration_config.json`.

### Fixed

- `WelfordMeanAccumulator` now defaults to `torch.float64` and tracks
  `AbliterationConfig.use_float64_subtraction`. Previously the
  accumulator hard-coded `float32`, silently truncating means before
  `compute_refusal_direction_float64` could promote them — the f64
  subtraction path was a no-op on the dominant code path.
- Outlier Winsorization and magnitude clipping now propagate into the
  Welford means consumed downstream. Previously, with the default
  `use_welford_mean=True`, clipping was applied to a list the refusal
  direction never read, while the quality scores and harmless boundary
  signals DID read it — leaving the three signals internally
  inconsistent. Both clipping helpers also now upcast to float64
  internally per the precision policy and downcast back to the input
  dtype on return.
- `apply_per_neuron_norm_preserving_projection`'s output-space branch
  (e.g., `o_proj` / `down_proj`) now decomposes per-row instead of
  per-column so per-output-neuron magnitudes are preserved as
  grimjim's [norm-preserving biprojected
  abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)
  sample requires. The pre-fix kernel preserved per-input-neuron norms
  — exactly opposite to the cited reference.
- Square attention projections (`hidden_size == num_heads * head_dim`)
  no longer fall through to the wrong projection axis. Both projection
  helpers accept an explicit `direction_space` override, and
  `abliterate_model` resolves it from the layer name via the new
  `infer_direction_space` helper, forcing the output-space branch for
  `o_proj` / `down_proj` / `lm_head` / `out_proj` / `c_proj` / `fc2`
  / `w2`. Pre-fix, the legacy shape heuristic always took the
  input-axis branch on those, silently corrupting `o_proj` semantics.
- Cross-layer biprojection now matches grimjim's per-L formulation
  instead of the layer-ensemble mean it was masquerading as. New
  `compute_biprojected_direction` helper implements
  `r_bi(M, L) = r_M - (r_M * h_hat_L) * h_hat_L` with the optional
  two-pass step, and `compute_biprojection_mapping` resolves each
  intervention layer `L` to its source measurement layer `M` via a
  `"nearest"` (default), `"single"`, or explicit-dict policy
  (`AbliterationConfig.biprojection_mapping`). Harmless means at
  intervention layers are now always computed when biprojection is
  enabled. The legacy ensemble path is preserved behind
  `AbliterationConfig.use_direction_ensemble` for opt-in
  reproducibility. Resolved policy and `L -> M` map round-trip into
  `abliteration_config.json` for audit.
- `AbliterationConfig.min_quality_threshold` is now wired into
  `select_biprojection_layers` to drop low-quality candidates before
  the top-N ranking. Previously the field had no reader anywhere in
  the repo.

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
