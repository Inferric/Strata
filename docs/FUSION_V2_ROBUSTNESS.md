# Fusion v2 rolling robustness protocol

Status: frozen before full-run metrics  
Frozen: 2026-07-30 19:10 America/Chicago  
Parent protocol: `docs/FUSION_V2_PROGRAM.md`

## Evidence from the completed screen

All 24 preregistered fold-1/seed-17 screen arms completed. The short-context
Fusion v2 had the lowest custom point RMSE but failed the tail-regression rule.
The all-scale concatenation model was the best Fusion v2 candidate satisfying
the screen advancement tolerances, although its 0.97% improvement over
persistence did not meet the immutable 2% final success threshold. Raw learned
Student-t scale was better calibrated than the pooled conformal alternatives.
The exact screen identities and selection calculation are frozen in
`research/experiments/fusion-v2-program/screen-freeze.json`.

## Frozen hypothesis

The constraint-respecting Fusion v2 configuration will retain a lower primary
RMSE than persistence across three chronological weather periods and seeds 17,
41, and 73 without material tail, bias, CRPS, coverage, or stability
regression. This test characterizes robustness; it does not retroactively turn
the screen into success evidence.

## Frozen candidate

- candidate: `selected-fusion-v2`;
- 192 hidden units, six heads, four experts, and about 3.61M parameters;
- short, medium, and slow Cn²/weather contexts;
- persistence-anchored residual and Fourier horizon conditioning;
- concatenative deep weather interaction;
- full physical/refractivity tokens;
- no atmospheric-pretraining initialization, because both screened pretraining
  recipes were negative;
- raw learned Student-t scale and quantiles, because pooled conformal
  calibration degraded coverage distance and CRPS on identical point
  forecasts.

This is a custom Fusion v2 candidate, not a champion or promotion candidate.

## Rolling evidence matrix

The exact frozen rolling folds remain unchanged. The candidate runs on folds 1,
2, and 3 for seeds 17, 41, and 73. Each run uses up to 50,000 chronological
training examples, up to 12,000 selection examples, BF16, at most 12 epochs,
selection-only early stopping, and fold-calibration-only uncertainty handling.

Controls are:

- persistence, climatology, and diagnostic LightGBM once per fold;
- MLP, TCN, and Horizon v1 on every fold and all three seeds.

All models use the same endpoint identities, horizons, preprocessing fit, and
resource envelope. LightGBM remains diagnostic only.

## Frozen analysis

The primary endpoint is the mean RMSE across 15, 30, and 60 minutes. The
5-minute horizon remains an anchor. Report fold/seed direction, pooled and
per-horizon RMSE, MAE, bias, tail MAE, CRPS, Student-t NLL, 80% coverage and
width, selective/OOD risk, runtime, board/process VRAM, RAM, and artifact
bytes.

The immutable final conditions remain:

1. at least 2% primary RMSE improvement over persistence and the stronger of
   MLP/TCN;
2. a positive paired 24-hour block-bootstrap 95% interval with 2,000
   resamples and seed 20260730;
3. the same improvement direction for seeds 17, 41, and 73;
4. 80% coverage inside 70--90%;
5. no material bias, tail MAE, CRPS, or stability regression;
6. provenance, leakage, reproducibility, resource, and artifact checks pass.

Evaluated conditions are `PASS` or `FAIL`; unavailable conditions are
`NOT_EVALUATED`.

## Confirmation rule

The March--April 2022 confirmation labels remain sealed during this matrix. A
confirmation release is allowed only if the already frozen rolling evidence
makes the candidate eligible under every condition above, the candidate and
calibration identities are committed, and the explicit full-fit
`--release-confirmation` path is used. Otherwise confirmation stays untouched
and is reported as `NOT_EVALUATED`.

## Safety and deadline

One local training process, zero workers, four CPU threads, zero cloud dollars,
and the existing RAM/VRAM/temperature/storage stop rules remain immutable. No
new run starts after 2026-07-31 06:06 America/Chicago. Remaining neural
wall-clock time is bounded by the program's eight GPU-hour ceiling.
