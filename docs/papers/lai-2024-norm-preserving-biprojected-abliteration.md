---
title: Norm-Preserving Biprojected Abliteration
author: Jim Lai (grimjim)
source: https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
published: 2025-11-06
retrieved: 2026-04-22
license: "Copyright of the original author. Archived here under fair use for
  academic reference; replace or remove if redistribution is restricted."
---

# Norm-Preserving Biprojected Abliteration

Abliteration is a technique for removing refusal behaviors from language models
by identifying and intervening on "refusal directions" in activation space,
notionally represented via a single mean refusal direction. This finding has
been useful in mechanistic interpretability.

We recently presented a refinement called "projected abliteration" that
improves upon the conventional approach by removing only the mechanistically
relevant components of the refusal direction, confirming a prior finding that
LLMs encode refusal and harmfulness separately. After that, we further refined
the technique to "biprojected abliteration", which also removed the
corresponding component when removing refusal measured using one layer from
another layer entirely; in principle this would avoid disturbing the harmless
direction of any layer targeted for intervention. Interestingly, some safety
refusal came back.

Upon further consideration, there remained an issue with regard to layer
weight modification and weight norms.

In conventional (and our previously modified) abliteration, the normalized
refusal direction is subtracted from the layer's residual streams targeted for
intervention, specifically `self_attn.o_proj` and `mlp.down_proj`. Although
this is effective in practice for steering, it is mathematically unprincipled
because:

- the direction removed contains a unit magnitude component in addition to the
  directional component, complicating interpretation,
- the removal does not respect the relative importance of neurons, resulting
  in unpredictable scale effects, and
- disturbs the geometry of the weight matrix in unpredictable ways.

Contrary to conventional wisdom that abliteration significantly degrades model
capabilities, our norm-preserving approach improved reasoning performance over
the baseline model (NatInt: 21.33 vs 18.72), while achieving effective refusal
removal (UGI: 32.61 vs 19.58).

## A Refined Mathematical Intervention

Instead of subtracting the refusal direction from the target weights, we
propose subtracting only the directional component while preserving the norm
of the weights.

Applying norm-preservation is more respectful of existing layer normalization
by maintaining the relative activation scale structure that the model's
normalization layers were trained to expect. We should therefore expect some
improvement over naive abliteration with regard to reducing incidental damage
to reasoning. Furthermore, the ablation can still be performed as a rank-1
modification, keeping the overall approach computationally efficient.

Given:

- Weight matrix \( \mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \)
- Refusal direction \( \mathbf{r} \in \mathbb{R}^{d_{\text{out}}} \) (refined via biprojection)
- Scaling factor \( \alpha \in [0, 1] \)

**Step 1: Normalize the refusal direction**

\[ \hat{\mathbf{r}} = \frac{\mathbf{r}}{\|\mathbf{r}\|_2} \]

**Step 2: Decompose weight matrix into magnitude and direction**

For each row \( i \) of \( \mathbf{W} \):

\[ m_i = \|\mathbf{W}_{i,:}\|_2 \qquad \hat{\mathbf{W}}_{i,:} = \frac{\mathbf{W}_{i,:}}{\|\mathbf{W}_{i,:}\|_2} \]

Or in matrix form:

\[ \mathbf{M} = \text{diag}(\|\mathbf{W}_{1,:}\|_2, \ldots, \|\mathbf{W}_{d_{\text{out}},:}\|_2) \qquad \hat{\mathbf{W}} = \mathbf{M}^{-1}\mathbf{W} \]

**Step 3: Ablate refusal direction from the normalized directional component**

Compute projection coefficients (alignment of each input dimension with
refusal):

\[ \mathbf{p} = \hat{\mathbf{r}}^T \hat{\mathbf{W}} \in \mathbb{R}^{d_{\text{in}}} \]

Remove the refusal component via rank-1 update:

\[ \hat{\mathbf{W}}_{\text{ablated}} = \hat{\mathbf{W}} - \alpha \cdot \hat{\mathbf{r}} \mathbf{p}^T \]

Renormalize each row to unit length, to enable recombination with original
magnitudes:

\[ \hat{\mathbf{W}}_{\text{new}} = \text{normalize}(\hat{\mathbf{W}}_{\text{ablated}}, \text{dim}=1) \]

**Step 4: Recombine with original magnitudes**

\[ \mathbf{W}_{\text{new}} = \mathbf{M}\,\hat{\mathbf{W}}_{\text{new}} \]

This ensures that \( \|\mathbf{W}_{\text{new}, i,:}\|_2 = \|\mathbf{W}_{i,:}\|_2 \) for all rows \( i \), preserving the learned importance structure while
redirecting computation away from the refusal direction.

## Sample PyTorch implementation

```python
import torch

# Core implementation (excerpt from complete function)
"""
Args:
    W: Weight matrix of shape [out_features, in_features]
    refusal_dir: Refusal direction vector of shape [out_features]
    scale_factor: Scaling factor for ablation strength (default: 1.0)
"""

# Normalize refusal direction
refusal_normalized = torch.nn.functional.normalize(refusal_dir, dim=0)

# Decompose weight matrix
W_norm = torch.norm(W, dim=1, keepdim=True)  # [out_features, 1]
W_direction = torch.nn.functional.normalize(W, dim=1)  # normalized per output neuron

# Apply abliteration to the DIRECTIONAL component
projection = torch.matmul(refusal_normalized, W_direction)  # [in_features]
W_direction_new = W_direction - scale_factor * torch.outer(refusal_normalized, projection)

# Re-normalize the adjusted direction to enable recombination
W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)

# Recombine: keep original magnitude, use new direction
W_new = W_norm * W_direction_new
```

## Layer Selection Methodology

We measured refusal directions across all layers but required a principled
approach to select which measurements to use for intervention. Our selection
employed a composite quality metric combining three factors:

**Signal-to-noise ratio (SNR):** the magnitude of the refusal direction
relative to the mean activations:

```
snr = ||r|| / max(||harmful_mean||, ||harmless_mean||)
```

where the refusal direction \( \mathbf{r} = \text{harmful\_mean} - \text{harmless\_mean} \).

**Cosine dissimilarity:** the angular separation between harmful and harmless
activation means:

```
dissimilarity = 1 - cosine_similarity(harmful_mean, harmless_mean)
```

Higher values indicate more distinct representational geometry.

**Composite quality score:**

```
quality = snr × (1 - cos_sim)
```

By charting these metrics across all layers, we selected candidates exhibiting
both high SNR and strong cosine dissimilarity, with particular attention to
layers showing sharp changes in these metrics. The suitability of a selected
refusal direction for application to nearby and preceding layers was informed
by tracking the cosine similarity evolution of refusal directions across
consecutive layers — stable directional alignment suggested robust cross-layer
applicability.

This approach is admittedly heuristic, but grounded in the observed geometric
structure of refusal representations. Future work might formalize optimal
layer selection through systematic ablation studies.

This heuristic is computationally efficient: a single inference pass captures
activations across all layers, and quality metrics are computed post-hoc from
this data. Unlike iterative search approaches that require multiple model
evaluations, the analysis adds negligible overhead to the standard abliteration
workflow, merely extracting more signal from measurements already performed.

For Gemma3 12B Instruct, with layers numbered `[0..47]`, we picked the
measurements from layers 23 and 29 for broad application. Keeping the refusal
and mean harmless direction measurements proved to be vital in subsequent
refinements.

## Result

With this revised method, we abliterated Gemma3 12B Instruct yet again. As
before, we applied a default scale factor of 1.0, intervening on layers
`[11..41]`. As expected, we were able to bypass refusal with harmful test
prompts. The model retained more of its capabilities in informal testing, and
`grimjim/gemma-3-12b-it-norm-preserved-biprojected-abliterated` scored highest
on the UGI and NatInt benchmarks on the
[UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)
compared to our prior published abliteration variants for the same baseline
model, and the baseline Instruct model itself.

As before, during measurement of activations, magnitude sparsification at
0.995 strength was applied when obtaining measurements from prompts. This was
necessary to distinguish the refusal direction between the mean harmful and
harmless directions. The empirical observation was that strong outlier
activations characterize the model.

To maximize numerical stability, we continued to employ 32-bit floating point
throughout for intermediate calculations even though the model was released in
16-bit bfloat16 floating point format. It was previously noted that performing
intermediate calculations in 16-bit bfloat16 led to suboptimal results. We
recommend that at least 32-bit floating point be employed in models which
evince a large variance in activation magnitudes.

## Discussion

By successfully narrowing down the intervention to only the directional
component with preserved norms, we establish that refusal direction alone is
critical to abliteration outcomes, rather than refusal direction entangled
with magnitude effects. However, despite this theoretical grounding, it
remains probable that removing even the directional component entangled with
harmfulness assessment could reduce safety in unprincipled ways. Korznikov et
al. (2025) demonstrated that activation steering on even benign features can
compromise LLM safety, suggesting that interventions in representational space
may have unintended consequences for safety mechanisms.

Preserving the magnitude was likely important in the case of
[Gemma3 12B Instruct](https://huggingface.co/google/gemma-3-12b-it), given
that high magnitude outliers obscured the underlying refusal direction almost
certainly encode important behavioral information that should be preserved in
order to retain functionality, as reported by Sun et al. (2024).

Benchmark results on the
[UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)
demonstrated clear improvements over prior abliteration variants:

| Model Variant                      | UGI Score | NatInt Score |
| ---------------------------------- | --------- | ------------ |
| Gemma-3 12B Instruct (baseline)    | 19.58     | 18.72        |
| Standard abliterated               | 32.08     | 18.64        |
| Norm-preserved biprojected         | 32.61     | 21.33        |

Notably, while standard abliteration achieved comparable uncensoring (UGI
scores), it showed slight capability degradation (NatInt: 18.64 vs baseline
18.72). The norm-preserving approach not only matched the uncensoring
effectiveness but significantly improved reasoning capability (NatInt: 21.33).
This finding aligns with recent observations of a 'Safety Tax' phenomenon
(Huang et al., 2025), where safety alignment has been shown to degrade
reasoning capabilities in language models. The improvement over baseline
suggests that removing directionally-encoded safety constraints may unlock
latent reasoning capabilities that were suppressed by safety mechanisms,
though this relationship warrants further investigation.

Although we had previously empirically established that intervention on
multiple layers was required to achieve the desired rate of compliance to
harmful prompts, we found theoretical grounding in a 2023 paper by McGrath et
al. titled "The Hydra Effect: Emergent Self-repair in Language Model
Computations". The authors demonstrated that when individual layers are
ablated, other layers adaptively compensate to restore approximately 70% of
the original computation. This self-repair mechanism explains why
single-layer interventions generally prove insufficient for robust
abliteration, as the model inherently routes around localized damage.

A multi-layer intervention strategy directly addresses this challenge: by
simultaneously modifying both attention output projections and MLP down
projections across multiple layers, one can effectively "cut multiple heads
of the hydra" at the same time, preventing the compensatory mechanisms from
restoring refusal behavior. Via judicious selection of layer measurements
and a set of intervention layers, a structured rank-2L intervention (where
L is the number of targeted layers) can provide sufficient coverage to
overcome emergent self-repair while remaining computationally efficient
through localized rank-1 updates at each weight matrix. This "hydra effect"
in retrospect accounted for the partial return of safety refusals during
"biprojected abliteration".

We restructured the traditional abliteration pipeline into three distinct
phases: (1) measurement across all layers, (2) analytical layer selection via
quality metrics, and (3) targeted intervention on selected layers. This
separation provides flexibility: rather than committing to a single 'best'
layer, practitioners can select multiple high-quality candidates for
intervention, enabling the multi-layer strategy necessary to overcome the
hydra effect while maintaining computational efficiency through the rank-1
structure of each individual weight modification.

Finally, an interesting practical consequence emerges from this understanding:
the number of intervention layers provides a crude but effective mechanism for
regulating the compliance-safety tradeoff. Fewer layers allow more self-repair,
preserving some refusal capability, while more layers overcome the
compensatory mechanisms more thoroughly. This provides practitioners with a
tunable parameter for calibrating model behavior according to their use case
and risk tolerance.

## References

- Arditi et al. — *Refusal in LLMs is mediated by a single direction*, LessWrong (2024). <https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction>
- Huang et al. — *Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable*, arXiv preprint (2025). <https://arxiv.org/abs/2503.00555>
- Korznikov et al. — *The Rogue Scalpel: Activation Steering Compromises LLM Safety*, arXiv preprint (2025). <https://arxiv.org/abs/2509.22067>
- Labonne — *Uncensor any LLM with abliteration*, huggingface.co (2024). <https://huggingface.co/blog/mlabonne/abliteration>
- Lai — *Projected Abliteration*, HuggingFace article (2025). <https://huggingface.co/blog/grimjim/projected-abliteration>
- McGrath et al. — *The Hydra Effect: Emergent Self-repair in Language Model Computations*, arXiv preprint (2023). <https://arxiv.org/abs/2307.15771>
- Sun et al. — *Massive Activations in Large Language Models*, arXiv preprint (2024). <https://arxiv.org/abs/2402.17762v2>
- Zhao et al. — *LLMs Encode Harmfulness and Refusal Separately*, arXiv preprint (2025). <https://arxiv.org/abs/2507.11878>

---

*Initial publication date: November 6, 2025.*
*Archived snapshot retrieved 2026-04-22 from the source URL above. Minor
formatting (LaTeX math, code fencing) cleaned up for offline readability;
content otherwise verbatim. Comments from the Hugging Face discussion thread
are not included — see the live article for the comment thread.*
