# Fusion v2.8 horizon-conditioned bilinear shortcut cycle

Status: **frozen 2026-07-31 01:27 America/Chicago; no v2.8 arm has run**

Parent: `docs/FUSION_V27_RESIDUAL_CAP.md`

Parent evidence run: `5164a7dec2854bb28439ced00b78227e`

Evaluation role: rolling development selection only

## Frozen parent evidence and architectural audit

Fusion v2.5's fixed history-linear shortcut produced the program's strongest
custom-model rolling result, but it missed the 2% MLP margin and tail-MAE gate.
Training-tail reweighting (v2.6) and residual bounding (v2.7) each produced
three non-improvements and are closed. These failures show that neither
target emphasis nor rare output magnitude alone explains the remaining gap.

The selected shortcut concatenates a 192-wide Fourier horizon embedding with
126 multiscale history values and applies one linear map:

\[
\delta_{\mathrm{shortcut}} =
  w_h^\top h + w_x^\top x + b.
\]

Because there is no multiplicative term, \(w_x\) is identical at 5, 15, 30,
and 60 minutes. Horizon features can change only an additive bias. The deep
weather-conditioned branch is horizon-aware, but its direct autoregressive
path is not. This code-level audit was completed without viewing a new metric
or any sealed role.

## Evidence-motivated question

This cycle asks:

> Can a small horizon-conditioned interaction let the direct multiscale
> history shortcut use different autoregressive corrections at different
> forecast leads, improving both point and tail error without enlarging the
> deep backbone?

This is a distinct architecture family. It does not revisit the closed loss,
residual-cap, residual-exponent, or generic shortcut-input searches.

## Frozen architecture

Retain the exact Fusion v2.5 deep branch, inputs, probability head, optimizer,
nine-epoch cap, and additive history-linear shortcut. Add one zero-initialized
low-rank interaction:

\[
\delta_{\mathrm{shortcut}} =
  w^\top[h;x] + b
  + v^\top\left[
      \tanh(Ax)\odot\tanh(Bh)
    \right],
\]

where:

- \(x\) is the 126-value short/medium/slow history representation already
  used by the shortcut;
- \(h\) is the 192-wide Fourier horizon embedding;
- \(A\in\mathbb R^{r\times126}\),
  \(B\in\mathbb R^{r\times192}\), and \(v\in\mathbb R^r\);
- \(w\), \(b\), and \(v\) start at zero, preserving the persistence anchor;
- rank \(r\) is the only intervention variable.

The additive term exactly preserves the current shortcut's function class.
The bilinear term adds \(319r\) parameters:

| candidate | interaction rank | expected parameters |
|---|---:|---:|
| `horizon-bilinear-control` | 0 (history-linear control) | 3,611,849 |
| `horizon-bilinear-r4` | 4 | 3,613,125 |
| `horizon-bilinear-r8` | 8 | 3,614,401 |
| `horizon-bilinear-r16` | 16 | 3,616,953 |

All remain inside the frozen 2--8M local parameter target. No rank
interpolation, nonlinear shortcut depth, extra weather shortcut, new input,
or post-metric initialization change is permitted.

All other settings remain fixed: concatenative weather interaction in the
deep multiscale branch, full physics tokens, no pretraining, residual exponent
zero, unbounded residual, no generic or tail point loss, raw Student-\(t\)
scale, ordered quantiles, 12,000 training sequences, 4,000 calibration
examples, 4,000 selection examples, fold 1, seed 17, BF16, batch size 128,
accumulation 2, learning rate \(3\times10^{-4}\), and weight decay 0.01.

## Frozen screen and decision

Run all four arms sequentially through Prefect and MLflow at one committed
revision. The rank-zero control reruns the exact v2.5 history-linear shortcut.

An intervention advances only if all of the following hold versus the paired
rank-zero control:

1. primary mean RMSE over 15/30/60 minutes improves by at least 0.75%;
2. primary tail MAE improves by at least 0.5%;
3. CRPS does not regress by more than 2%;
4. 80% interval coverage remains in 70--90%;
5. absolute bias is at most 0.25 log10 \(C_n^2\);
6. metrics are finite and provenance, exact parameter delta, zero-init
   identity, residual components, artifacts, and resources pass.

The 0.75% screen threshold is the evidence-derived minimum needed to close the
rolling MLP gap with a small safety allowance; the immutable full 2% success
margin is unchanged.

If multiple arms advance, select the lowest primary RMSE; a tie within 0.25%
selects the lower rank. Freeze the exact winner for nine fresh full-data runs
over folds 1--3 and seeds 17, 41, and 73. Full success still requires at least
2% over persistence and the stronger neural control, positive paired 24-hour
bootstrap intervals (2,000 resamples; seed 20260730), the same direction for
all seeds, 70--90% coverage, and no material bias, tail, CRPS, or stability
regression.

If none advances, the three preregistered ranks close this horizon-bilinear
family. Do not add another rank or interaction after metrics.

## Leakage, safety, and budget

- Use only the verified public USNA-large snapshot and frozen purged rolling
  split. Never edit raw or sealed data.
- Fit preprocessing on fold training only; use calibration only for
  uncertainty records and selection only for checkpointing/this screen.
- Never load USNA confirmation labels or the official MLO test.
- Validate one autonomous proposal contract per positive rank.
- Use one local training process, zero workers, four CPU threads, BF16, and
  zero cloud dollars.
- Retain all RAM, VRAM, temperature, storage, OOM-recovery, and unrelated
  process protections.
- Each arm is capped at 0.5 GPU-hours; the screen is capped at 2.0 GPU-hours
  and remains within the eight-hour program ceiling.
- Start no run after 2026-07-31 06:06 America/Chicago.

## Required evidence

Every run records exact repository/data/split identities, rank and parameter
count, initialization contract, checkpoint, predictions, additive/deep/final
residual diagnostics, calibration, epochs and optimizer steps, runtime,
board/process VRAM, RAM, artifact bytes, failures, and retry state. The cycle
ends with exact run identities, immutable decisions, a ledger event, MLflow
evidence, and a compiled and visually inspected LaTeX report.
