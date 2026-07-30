# Cn² modeling research brief

The full public-source briefing is preserved as LaTeX at
`research/Cn2_Advanced_Modeling_Research_Briefing.tex`. It contains the complete
decision brief, 32 annotated high-priority papers, 16 dataset cards, an
experiment backlog, and a 193-item discovery bibliography. The original
36-page DOCX is retained at
`research/archive/Cn2_Advanced_Modeling_Research_Briefing.docx` as a
layout-faithful archive; this file records the decisions that shape the
repository.

## Bottom line

Proceed, but treat the effort as a data-and-validation program with a neural
model at its center. Learning useful Cn² relationships is viable. Universal
generalization is not yet demonstrated because direct measurements are sparse,
heterogeneous, geometry-dependent, and concentrated in a few sites and
instruments.

The immediate publication-quality target is not a billion-parameter foundation
model. It is:

1. a reproducible geometry- and provenance-aware benchmark;
2. a compact probabilistic surface model that wins on sealed blocked-time tests;
3. a profile model pretrained on teacher fields and fine-tuned/evaluated on
   independent observations; and
4. calibrated uncertainty, tail/layer diagnostics, and explicit OOD behavior.

## Why Cn² remains difficult

Cn² spans orders of magnitude and represents unresolved refractive-index
fluctuations. Routine atmospheric fields do not resolve the relevant scales.
Boundary-layer transitions, terrain, moisture covariance, wakes, and land–sea
interfaces create sharp nonstationarity. Labels are not interchangeable:

- sonic-derived estimates are local and processing-dependent;
- scintillometers and optical links are path weighted;
- MASS, SCIDAR, SLODAR, and thermosondes have different vertical responses;
- seeing, Fried parameter, isoplanatic angle, coherence time, and scintillation
  are weighted functionals of a profile; and
- WRF/LES/parameterization outputs are teachers, not independent truth.

Random row splitting, global normalization, or mixing these label classes
without observation geometry can generate impressive but invalid results.

## Evidence from the literature

- `otbench` demonstrates the value of operational tasks and blocked evaluation.
- OTProf establishes a strong deep profile precedent with a compact
  convolution-attention model, while also documenting smoothing of rare strong
  layers.
- Few-shot Mauna Loa work indicates useful transfer from pretrained tabular
  representations under label scarcity.
- Physics-informed video estimation transfers better than a pure image
  regressor across scenes.
- Π-ML and modern parameterization comparisons show that dimensionless,
  physically meaningful variables remain strong baselines.
- OTCliM demonstrates multi-year, multi-station potential while exposing urban
  and site-shift failures.
- Neural operators support continuous coordinate queries; residual generative
  modeling is a credible later mechanism for restoring sharp stochastic layers.

No cited public work closes the full gap: continuous coordinates, typed
observation operators, latent thermodynamic covariance, soft regime experts,
calibrated extremes, and multisite evaluation.

## Model-family decision

The full research concept is **Strata-OT**, a continuous probabilistic
atmospheric operator. This repository starts with two earned stages:

- **Strata-OT Surface** (2–8M parameters): a temporal transformer, soft
  regime mixture, physical-baseline residual, Gaussian location/scale, and
  ordered quantiles. This is the first direct-label Mauna Loa model.
- **Strata-OT Column** (8–15M parameters): pressure-coordinate Fourier
  features, local vertical convolutions, Transformer encoding, cross-attention
  queries on arbitrary output levels, regime routing, and probabilistic heads.

Only after these stages pass cross-time/site/instrument gates should the project
add multiscale NWP fields, typed observation tokens, explicit thermodynamic
covariance, differentiable observation operators, or a flow/diffusion residual.

## Public-release boundary

The repository is an atmospheric-effects research core. Controlled datasets,
mission adapters, weapon-effects integration, target vulnerability, engagement
logic, and operational performance claims remain outside it.
