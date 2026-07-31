# Fusion v2.7 bounded-residual architecture cycle

Status: **frozen 2026-07-31 01:10 America/Chicago; no v2.7 arm has run**

Parent: `docs/FUSION_V26_TAIL_OBJECTIVE.md`

Parent evidence run: `f6993c836f6f4708aa340ac5d989a861`

Evaluation role: rolling development selection only

## Frozen parent evidence and diagnostic

Fusion v2.5 improved rolling primary RMSE over persistence, MLP, LightGBM, and
Fusion v2, but missed the immutable 2% MLP margin and tail-MAE gate. V2.6 then
applied a training-only upper-decile Huber term to the fixed architecture.
None of its three weights advanced: the strongest point result improved RMSE
by only 0.49%, and every intervention worsened tail MAE by 7.5--17.3%.
That objective family is closed.

Before freezing this cycle, residual magnitudes—not target errors—were audited
from the nine already-completed v2.5 rolling runs. Across 108,000 logged
development forecasts, the learned correction had:

- median absolute magnitude 0.0476 log10 units;
- 95th absolute percentile 0.2009;
- 99th absolute percentile 0.3348;
- 99.9th absolute percentile 0.5841;
- observed range \([-1.4526, 0.8079]\).

This target-free diagnostic shows rare large corrections that can defeat the
safety of a persistence anchor. It motivates the fixed bounds below but does
not reveal their forecast metrics.

## Evidence-motivated question

This cycle asks:

> Does explicitly bounding the learned correction preserve useful small
> deviations from persistence while preventing rare corrections from
> degrading point and tail error?

This is a distinct output-architecture family. It does not change the loss,
training target distribution, weather inputs, probability distribution, or
closed horizon-scaling exponent family.

## Frozen intervention

Keep the exact 3,611,849-parameter v2.5 architecture, nine-epoch cap, data
roles, and optimizer. Let

\[
r_{\mathrm{raw}} =
  \delta_{\mathrm{deep}}+\delta_{\mathrm{history\ shortcut}}.
\]

The existing exponent-zero model uses
\(\hat y=y_{\mathrm{latest}}+r_{\mathrm{raw}}\). The intervention replaces only
the output correction with a symmetric hard bound:

\[
\hat y =
  y_{\mathrm{latest}}+
  \operatorname{clamp}(r_{\mathrm{raw}},-c,c).
\]

The same bounded location anchors the Student-\(t\) and ordered-quantile heads.
The raw and final corrections are both logged. Exact arms:

| candidate | bound \(c\), log10 \(C_n^2\) |
|---|---:|
| `residual-cap-unbounded` | 0.00 (paired unbounded control) |
| `residual-cap-0p20` | 0.20 |
| `residual-cap-0p35` | 0.35 |
| `residual-cap-0p60` | 0.60 |

The three positive bounds bracket the completed residual audit's 95th, 99th,
and 99.9th absolute percentiles. A value of zero means no clamp; it does not
force a zero residual. No cap interpolation, asymmetric bound, soft transform,
or horizon-specific bound may be added after metrics are viewed.

All other settings remain fixed: short/medium/slow histories, concatenative
weather interaction, full physics tokens, history-linear shortcut, residual
horizon exponent zero, no pretraining, no generic or tail point loss, raw
Student-\(t\) scale, 12,000 training sequences, 4,000 calibration examples,
4,000 selection examples, fold 1, seed 17, BF16, batch size 128,
accumulation 2, learning rate \(3\times10^{-4}\), and weight decay 0.01.

## Frozen screen and decision

Run all four arms through Prefect and MLflow at one committed revision. The
unbounded control proves that `residual_cap=0` preserves the parent
architecture.

An intervention advances only if all of the following hold versus the paired
unbounded control:

1. primary mean RMSE over 15/30/60 minutes improves by at least 0.75%;
2. primary tail MAE improves by at least 0.5%;
3. CRPS does not regress by more than 2%;
4. 80% interval coverage remains in 70--90%;
5. absolute bias is at most 0.25 log10 \(C_n^2\);
6. metrics are finite and provenance, cap identity, raw/final residual
   diagnostics, artifacts, and resources pass.

The 0.75% advancement rule remains the evidence-derived minimum needed to
close v2.5's rolling MLP margin with a small safety allowance. It does not
change the immutable full success rule.

If one or more arms advance, select lowest primary RMSE, breaking a tie within
0.25% toward the least restrictive (largest) bound. Freeze it unchanged for
all three rolling folds and seeds 17, 41, and 73. Full success still requires
at least 2% over persistence and the stronger neural control, both positive
paired 24-hour bootstrap intervals (2,000 resamples; seed 20260730), same
direction for all seeds, 70--90% coverage, and no material bias, tail, CRPS,
or stability regression.

If none advances, the three preregistered bounds close this bounded-residual
family. Do not add another bound after selection metrics.

## Leakage, safety, and budget

- Use only the verified public USNA-large snapshot and frozen purged rolling
  split. Never edit raw or sealed data.
- Fit preprocessing on fold training only; use calibration only for
  uncertainty records and selection only for checkpointing/this screen.
- Never load USNA confirmation labels or the official MLO test.
- Validate one autonomous proposal contract per positive bound.
- Use one local process, zero workers, four CPU threads, BF16, and zero cloud
  dollars.
- Retain all RAM, VRAM, temperature, storage, OOM-recovery, and
  unrelated-process protections.
- Each arm is capped at 0.5 GPU-hours; the screen is capped at 2.0 GPU-hours
  and stays inside the eight-hour program ceiling.
- Start no run after 2026-07-31 06:06 America/Chicago.

## Required evidence

Every run records the exact repository/data/split identities, requested bound,
raw and final residual distributions, checkpoint, predictions, component
diagnostics, calibration, epochs and optimizer steps, runtime, board/process
VRAM, RAM, artifact bytes, failures, and retry state. The cycle ends with
exact run identities, immutable decisions, a ledger event, MLflow evidence,
and a compiled and visually inspected LaTeX report.
