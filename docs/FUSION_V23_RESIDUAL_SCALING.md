# Fusion v2.3 horizon-scaled residual cycle

Status: **frozen 2026-07-30 22:50 America/Chicago; no residual-scaling arm has run**

Parent: `docs/FUSION_V22_DATA_SCALE.md`

Parent evidence run: `c8685b9bea5d4ad391e50045809d51d1`

Evaluation role: rolling development selection only

## Evidence-motivated question

Fusion v2 predicts a single raw correction to persistence after receiving a
Fourier horizon embedding. In the completed rolling matrix, the standard
deviation of its realized residual contribution increased from about 0.039 at
5 minutes to 0.117 at 60 minutes. Explicit point-loss weights did not pass, and
denser use of 30,000, 50,000, or all 62,263 fold-1 sequences did not improve
the 12,000-sequence control. Those families are closed.

This cycle asks one different representation/training-target question:

> Does an explicit, fixed forecast-horizon scale on the persistence residual
> help the unchanged neural representation learn corrections whose expected
> magnitude grows with lead time?

## Frozen intervention

Let the network produce an interpretable raw residual \(\delta_\theta\). Change
only the final residual representation:

\[
  \widehat y_h = y_{\mathrm{latest}} +
  (h / 5)^\alpha \delta_\theta,
\]

where \(h\) is the forecast horizon in minutes and \(\alpha\) is fixed per arm.
The five-minute anchor always has scale one.

| candidate | exponent \(\alpha\) | scales at 5 / 15 / 30 / 60 minutes |
|---|---:|---|
| `residual-exp-0p00` | 0.00 | 1.000 / 1.000 / 1.000 / 1.000 |
| `residual-exp-0p25` | 0.25 | 1.000 / 1.316 / 1.565 / 1.861 |
| `residual-exp-0p50` | 0.50 | 1.000 / 1.732 / 2.449 / 3.464 |
| `residual-exp-1p00` | 1.00 | 1.000 / 3.000 / 6.000 / 12.000 |

The last arm represents constant-error-rate accumulation; the square-root arm
represents diffusion-like growth; the quarter-power arm is a conservative
intermediate. These are fixed inductive biases, not fitted physical laws.

Everything else is identical: 192 hidden units, six heads, four experts,
concatenative multiscale weather interaction, full physical tokens, no
pretraining, zero explicit point-loss weight, raw Student-t scale, 12,000
training sequences, 4,000 selection examples, fold 1, seed 17, BF16, three
epochs, batch size 128, accumulation 2, learning rate `3e-4`, and weight decay
`0.01`. The parameter count must remain exactly 3,611,530 for every arm.

Log the raw residual, deterministic residual scale, scaled residual
contribution, horizon attention, timescale weights, expert weights, weather
attention, modulation norm, and physics-token norm.

## Screen and decision

Rerun the zero-exponent control at the committed residual-scaling revision.
Compare all interventions against that paired control on primary mean RMSE over
15/30/60 minutes. An intervention advances only if:

1. primary RMSE improves by at least 2%;
2. tail MAE and CRPS do not regress by more than 2%;
3. 80% interval coverage remains in 70--90%;
4. absolute bias is at most 0.25 log10 Cn2;
5. metrics are finite and provenance, artifacts, parameter identity, and
   resources pass.

The five-minute result is a reported anchor. The lowest-RMSE
constraint-respecting arm advances to the unchanged three-fold, three-seed
protocol with its exact exponent. If none advances, the three
non-improvements close fixed power-law residual scaling. Do not add another
exponent after viewing results.

## Leakage, safety, and budget

- Use only the verified public USNA-large snapshot and frozen purged rolling
  split. Never edit raw or sealed data.
- Fit preprocessing on fold training only, use calibration only for
  uncertainty records, and decide on selection only.
- Never load USNA confirmation labels or the official MLO test.
- Validate all three intervention proposal contracts before execution.
- One local training process, zero workers, four CPU threads, BF16, no cloud,
  and the existing RAM/VRAM/temperature/storage stop rules.
- Each arm is capped at 0.5 local GPU-hours; the four-run cycle is capped at
  2.0 GPU-hours and remains inside the eight-hour program ceiling.
- No run begins after 2026-07-31 06:06 America/Chicago.

## Required evidence

Each run records the exponent, raw and scaled residual diagnostics, exact
code/data/split identities, checkpoint, predictions, calibration, runtime,
board/process VRAM, RAM, storage, failures, and retry state. The terminal
result requires a frozen identity map, MLflow evidence run, compiled and
visually inspected LaTeX PDF, ledger entry, and verified private Drive upload.
