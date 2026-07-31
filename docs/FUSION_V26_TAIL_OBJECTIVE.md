# Fusion v2.6 training-only tail-objective cycle

Status: **frozen 2026-07-31 00:32 America/Chicago; no v2.6 arm has run**

Parent: `docs/FUSION_V25_ROBUSTNESS.md`

Parent evidence run: `7af406bdddf94dd5a355bfbd46f7d94c`

Evaluation role: rolling development selection only

## Frozen parent evidence

Fusion v2.5 completed nine clean full-data runs across three chronological
folds and seeds 17, 41, and 73. It improved mean 15/30/60-minute RMSE by
6.51% over persistence, 1.32% over the stronger MLP, 12.09% over diagnostic
LightGBM, and 2.54% over Fusion v2. Both paired 24-hour bootstrap intervals
were positive and all seeds agreed.

The immutable result is nevertheless `FAIL`: the MLP margin was below the
required 2%, and aggregate tail MAE was 0.170982 versus the frozen
no-regression limit of 0.170385. Coverage, CRPS, bias, stability, provenance,
and resources passed. Confirmation labels remain sealed.

These two narrowly failed conditions motivate one distinct objective-level
question. They do not reopen the closed generic point-loss or history-shortcut
families and do not change any promotion threshold.

## Evidence-motivated question

The existing Student-\(t\) likelihood can accommodate large errors by changing
predictive scale, while the unweighted quantile term treats all target levels
equally. The parent model's only risk-metric failure occurs on the highest
observed target decile.

This cycle asks:

> Can a location loss applied only to high-\(C_n^2\) training targets reduce
> tail error enough to preserve the custom model's point-forecast advantage,
> without using selection statistics or changing its architecture?

This is not the v2.1 generic Huber family. V2.1 applied Smooth L1 to every
example on the no-shortcut architecture. V2.6 retains the fixed v2.5
history-linear architecture and applies the extra term only to targets above
a threshold fitted from that fold's training role.

## Frozen intervention

Keep the exact 3,611,849-parameter v2.5 architecture and policy:

- history-linear residual shortcut and unchanged deep residual branch;
- short, medium, and slow history/weather scales;
- concatenative weather interaction and full physics tokens;
- raw Student-\(t\) scale and ordered quantile head;
- no pretraining, residual-horizon exponent zero, and no generic point loss;
- BF16, batch size 128, accumulation 2, learning rate \(3\times10^{-4}\),
  weight decay 0.01, zero workers, and nine epochs maximum.

For each fold, construct the capped training sequence dataset first and fit

\[
q_{0.90}^{\mathrm{train}} =
  \operatorname{quantile}_{0.90}\{y_i : i\text{ is a training sequence}\}.
\]

The threshold uses direct log10 \(C_n^2\) training targets only. It is fixed
before optimization and recorded in the run manifest. Calibration, selection,
confirmation, and MLO test labels cannot enter it.

The existing forecast objective is

\[
\mathcal L_0 =
  \mathcal L_{\mathrm{Student}\text{-}t}
  + 0.20\,\mathcal L_{\mathrm{pinball}}.
\]

Add a masked Smooth L1 term with \(\beta=0.10\):

\[
\mathcal L_{\mathrm{tail}} =
  \frac{1}{B}\sum_{i=1}^{B}
  \mathbf 1[y_i \ge q_{0.90}^{\mathrm{train}}]\,
  \operatorname{SmoothL1}(\hat y_i-y_i;\beta=0.10),
\]

\[
\mathcal L = \mathcal L_0 + \lambda\mathcal L_{\mathrm{tail}}.
\]

The division remains by the full batch size, so the base gradient scale and
the observed approximately 10% training prevalence are preserved. The exact
frozen arms are:

| candidate | tail-Huber weight \(\lambda\) |
|---|---:|
| `tail-huber-0p00` | 0.00 (paired control) |
| `tail-huber-0p50` | 0.50 |
| `tail-huber-1p00` | 1.00 |
| `tail-huber-2p00` | 2.00 |

No weight, threshold quantile, beta, target direction, architecture, epoch
cap, or sample cap may change after metrics are viewed.

## Frozen screen and decision

Run all four arms on rolling fold 1, seed 17, with at most 12,000
chronologically sampled training sequences and 4,000 selection examples.
Run them through Prefect and MLflow at the same committed revision. The
zero-weight arm is rerun to prove the new implementation is inert at
\(\lambda=0\).

The full robustness result needs RMSE at most 98% of the stronger MLP:
0.246991. Relative to v2.5's 0.248705 rolling RMSE, that requires a 0.69%
additional reduction. To avoid advancing an effect smaller than the observed
gap, a screen intervention advances only if all of the following hold versus
the paired zero-weight control:

1. primary mean RMSE over 15/30/60 minutes improves by at least 0.75%;
2. tail MAE improves by at least 0.5%;
3. CRPS does not regress by more than 2%;
4. 80% interval coverage remains in 70--90%;
5. absolute bias is at most 0.25 log10 \(C_n^2\);
6. metrics are finite and provenance, threshold identity, artifacts,
   components, and resources pass.

The 0.75% screen rule is an advancement filter, not a changed success
criterion. If an arm advances, freeze the lowest-RMSE constraint-respecting
arm and evaluate nine fresh full-data runs across all three folds and seeds
17, 41, and 73. Full success retains every v2.5 rule: at least 2% over both
persistence and the stronger neural control, positive paired 24-hour
bootstrap intervals with 2,000 resamples and seed 20260730, same direction
for all seeds, 70--90% coverage, and no material bias, tail, CRPS, or stability
regression.

If no arm advances, the three distinct masked-tail weights close this
tail-objective family. Do not interpolate another weight or reuse selection
metrics to redefine the training threshold.

## Leakage, safety, and budget

- Use only the verified public USNA-large snapshot and frozen purged rolling
  split. Never edit raw or sealed data.
- Fit normalization and the tail threshold on fold training only.
- Use calibration only for uncertainty records and selection only for
  checkpointing and the frozen screen.
- Never load USNA confirmation labels or the official MLO test.
- Validate one autonomous proposal contract per intervention before execution.
- Use one local training process, `STRATA_NUM_WORKERS=0`, four CPU threads,
  BF16, and zero cloud dollars.
- Keep the existing RAM, VRAM, temperature, storage, OOM-recovery, and
  unrelated-process protections.
- Each arm is capped at 0.5 local GPU-hours; the four-run screen is capped at
  2.0 GPU-hours and remains inside the eight-hour program ceiling.
- Start no run after 2026-07-31 06:06 America/Chicago.

## Required evidence

Each run must record the exact repository/data/split identities, training-only
threshold and observed training tail fraction, objective weight, checkpoint,
predictions, component diagnostics, calibration, completed epochs and steps,
runtime, board/process VRAM, RAM, artifact bytes, failures, and retry state.
The cycle ends with immutable gate states, exact run identities, a ledger
event, MLflow evidence, and a compiled and visually inspected LaTeX report.
