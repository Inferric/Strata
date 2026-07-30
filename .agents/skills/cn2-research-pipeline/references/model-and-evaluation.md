# Model and evaluation

## Initial family

### Strata-OT Surface

Use for direct MLO/USNA scalar or path tasks on the 16 GB development GPU.

- Typed meteorological and geometry tokens
- Short temporal-context encoder
- Continuous stability/regime router
- Physics baseline plus learned residual
- Median, quantile, aleatoric-uncertainty, and OOD heads
- Target size: 2–8M parameters

### Strata-OT Column

Use for OTProf and later observed profiles.

- Per-pressure-level feature projection
- Pressure-aware vertical attention with relative height/pressure encoding
- Multiscale spectral/conv blocks for thin and broad layers
- Surface/context token and continuous regime routing
- Latent thermodynamic-turbulence state
- Differentiable refractivity decoder
- Quantile head plus optional generative residual
- Target size for local work: 8–15M parameters; scale only after data gates

## Required comparisons

- Persistence and climatology
- Hufnagel–Valley or task-appropriate physical parameterization
- ExtraTrees/LightGBM where available as diagnostic baselines
- Plain MLP or temporal convolution
- OTProf/Squeezeformer reproduction for profile work
- Current champion

Trees may be retained as baselines. They are not the target architecture.

## Primary splits

- blocked time
- leave-site and leave-campaign
- leave-instrument or observation geometry
- stable/neutral/convective and marine/urban/terrain regimes
- extreme/tail events

Group overlapping paths and colocated sensors so nearly identical atmosphere cannot cross train/test.

## Metrics

- Point/profile: log10 MAE, RMSE, bias, correlation, calibration slope
- Distribution: CRPS, NLL, interval coverage, sharpness
- Structure: layer detection F1, vertical displacement, peak error, tail recall
- Integrated optics: task-supported transforms such as seeing/r0/scintillation with uncertainty
- Operations: latency, peak VRAM, throughput, artifact size, cost, and selective risk under OOD rejection

## Promotion gate

Require:

1. reproducible data and code identities
2. no split/provenance violations
3. improvement on at least one primary out-of-distribution split
4. no material regression in calibration, tails, or integrated metrics
5. two or more seeds for a candidate, then a confirmatory sealed-test run
6. complete LaTeX report and model/data cards

Never promote from a tuning fold or a single lucky seed.
