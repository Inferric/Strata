# Strata-OT architecture

## Scientific target

Learn a calibrated distribution for \(\log_{10} C_n^2\) conditioned on
meteorology, history, surface state, location/vertical coordinate, and available
observations. Preserve a clear distinction between a direct prediction and a
prediction passed through a sensor/path observation operator.

## Stage 1: Strata-OT Surface

Input is a sequence \(\mathbf{x}_{t-k:t}\in\mathbb{R}^{k\times F}\).

1. Train-only standardized features are projected to width \(d\).
2. A pre-norm temporal Transformer learns transitions and delayed effects.
3. A differentiable router blends \(E\) residual MLP experts. Experts may
   specialize in stable, neutral, convective, maritime, terrain, or transition
   regimes without brittle labels.
4. Learned attention pooling summarizes the sequence.
5. The model predicts Gaussian location/scale and ordered 0.1/0.5/0.9
   quantiles in log space.
6. Where a defensible parameterization is available, the neural estimate is a
   residual around that physical baseline.

The initial loss is

\[
\mathcal{L} =
\operatorname{NLL}_{\mathcal N}(\log_{10}C_n^2)
+ 0.2\,\mathcal{L}_{\mathrm{pinball}}.
\]

The implementation lives in `src/strata_ot/models/strata_surface.py`.

## Stage 2: Strata-OT Column

For an atmospheric column at irregular pressure levels:

1. Embed log pressure with multiband Fourier coordinates.
2. Fuse coordinates and level features.
3. Apply a depthwise local convolution to protect thin vertical structure.
4. Encode the full column with axial self-attention.
5. Form query tokens at arbitrary output pressures.
6. Cross-attend from output coordinates to the encoded column.
7. Apply the same soft regimes and probabilistic heads at each output level.

This produces grid-flexible profiles without baking one output grid into the
weights. The implementation lives in
`src/strata_ot/models/strata_column.py`.

## Stage 3: heterogeneous observation learning

Add typed observation tokens and differentiable operators:

- point sampling with height/averaging support;
- path integration with instrument weighting;
- layered response matrices for MASS/SCIDAR/SLODAR;
- integrated \(r_0\), seeing, \(\theta_0\), and \(\tau_0\) conventions; and
- sensor/image encoders only when calibration metadata support them.

## Stage 4: latent physics and sharp residuals

Predict \(C_T^2\), \(C_q^2\), and \(C_{Tq}\) through a Cholesky-parameterized
temperature–humidity covariance so the variances are nonnegative and covariance
is bounded. A differentiable refractivity layer maps the state to
wavelength-aware \(C_n^2\). Only after deterministic profile evidence is strong,
train a conditional flow/diffusion model on residuals and vertical gradients to
recover sharp stochastic layers.

## Non-negotiable baselines

- climatology and persistence;
- a compact MLP;
- a physical/parameterization baseline when its required inputs exist;
- a tree model as a diagnostic comparator, never as the proposed architecture;
- OTProf reproduction for the teacher-profile task; and
- the previous champion.

Parameter matching, identical splits, and identical preprocessing are required.
