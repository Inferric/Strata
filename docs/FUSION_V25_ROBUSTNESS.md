# Fusion v2.5 rolling robustness protocol

Status: **frozen 2026-07-30 23:43 America/Chicago; no full-data v2.5 run has started**

Parent protocol: `docs/FUSION_V25_CONVERGENCE.md`

Parent evidence run: `dd8ab8627c2f4e418c6d23ed078fa7f7`

Evaluation role: rolling development selection only

## Frozen question

The nine-epoch history-linear shortcut passed the fold-1/seed-17 convergence
screen with a 4.07% primary RMSE improvement over its frozen no-shortcut
baseline, while tail MAE, CRPS, coverage, bias, provenance, artifacts, and
resources passed. This cycle asks:

> Does the exact selected candidate retain a material improvement across three
> chronological weather periods and seeds 17, 41, and 73?

This is a robustness test, not another architecture or duration search. No
width, input, shortcut, loss, calibration, epoch, data, or gate variant may be
introduced after rolling metrics are viewed.

## Frozen candidate and fresh run matrix

The candidate ID is `selected-fusion-v25`. It is the exact
3,611,849-parameter model selected in v2.5:

- short, medium, and slow past Cn²/weather histories;
- 192 hidden units, six attention heads, and four diagnostic experts;
- concatenative weather fusion and full physical/refractivity tokens;
- persistence anchor plus deep residual plus the history-only linear shortcut;
- no pretraining, explicit point loss, horizon residual scaling, or conformal
  post-hoc calibration;
- raw Student-t/quantile uncertainty heads;
- BF16, batch 128, accumulation 2, learning rate `3e-4`, weight decay `0.01`;
- maximum nine epochs with unchanged selection-only early stopping.

Run folds 1, 2, and 3 for seeds 17, 41, and 73: nine fresh neural runs. Each
uses at most 50,000 chronological training sequences and 12,000 selection
examples. Preprocessing fits only on the fold training role; uncertainty
records use only the fold calibration role.

## Exact control reuse

Do not retrain unchanged controls. Reuse only the exact 45 clean runs sealed in
`research/experiments/fusion-v2-program/robustness-freeze.json`:

- persistence, climatology, and diagnostic LightGBM once per fold;
- MLP, TCN, Horizon v1, and the preceding `selected-fusion-v2` across all
  folds and seeds.

The analyzer must load those exact run IDs and reject missing artifacts,
changed targets, timestamps, horizons, source hashes, split hashes, or dirty
provenance. This saves GPU time without changing any comparison endpoint.
LightGBM remains diagnostic only.

## Frozen decision

The primary endpoint is mean RMSE across 15/30/60 minutes; 5 minutes remains
an anchor. The v2.5 candidate is rolling-eligible only if all conditions pass:

1. at least 2% primary RMSE improvement over persistence and the stronger of
   the frozen MLP and TCN;
2. paired 24-hour block-bootstrap 95% intervals above zero versus both, using
   exactly 2,000 resamples and seed `20260730`;
3. positive direction versus persistence for seeds 17, 41, and 73;
4. mean 80% coverage inside 70--90%;
5. absolute bias at most 0.25 log10 Cn²;
6. tail MAE and CRPS no more than 2% worse than the stronger applicable
   frozen control;
7. relative fold/seed RMSE standard deviation at most 0.15;
8. exact provenance, pairing, components, artifacts, finite metrics, and
   resources pass.

Report its direction versus diagnostic LightGBM and the preceding custom
Fusion v2 honestly; neither comparison can weaken the frozen gates. If all
conditions pass, one confirmation release is permitted only after this exact
candidate, analysis, and calibration identity are committed, using the
explicit release interface. Otherwise confirmation remains sealed.

## Leakage, safety, budget, and stop conditions

- Use only the verified public USNA-large source and frozen purged rolling
  split. Never edit raw or sealed data.
- Never load the official MLO test or USNA confirmation labels during this
  matrix.
- One local training process, zero workers, and four CPU threads.
- Preserve all existing CPU/RAM/VRAM/temperature/storage start and stop
  thresholds. Permit one OOM recovery only.
- Each run is capped at 0.5 local GPU-hours; nine new runs are capped at 4.5
  hours and the full program remains below eight GPU-hours.
- Execute through five validated autonomy contracts containing 2+2+2+2+1
  runs; no contract exceeds the immutable two-run proposal ceiling.
- Cloud cost stays exactly zero.
- No new run starts after 2026-07-31 06:06 America/Chicago.
- Stop on leakage/provenance drift, corrupted labels, missing paired control
  artifacts, repeated numerical failure, failed OOM recovery, unsafe hardware,
  or inability to preserve exact evidence.

## Required terminal evidence

Freeze all nine new run IDs plus every reused control ID. Log exact configs,
epochs, optimizer steps, code/data/split hashes, checkpoints, predictions,
residual components, probabilistic metrics, paired bootstrap, runtime,
board/process VRAM, RAM, storage, failures, and retry state. Close the cycle
with MLflow evidence, a compiled and visually inspected LaTeX PDF, ledger and
Drive indexes, and a verified owner-only Drive upload. This development result
cannot autonomously promote a champion or merge the results branch.
