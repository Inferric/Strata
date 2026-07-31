# Fusion v2.4 direct residual-shortcut cycle

Status: **frozen 2026-07-30 23:05 America/Chicago; no shortcut arm has run**

Parent: `docs/FUSION_V23_RESIDUAL_SCALING.md`

Parent evidence run: `ab248fd98f0f47e8b5bcaca79fe70f21`

Evaluation role: rolling development selection only

## Evidence-motivated question

The complete rolling experiment found that Fusion v2 improved over persistence
but was 1.26% worse than the stronger MLP control. Explicit point loss, denser
within-fold sampling, and fixed horizon scaling all failed their frozen
screens. The custom model compresses each temporal branch through causal
blocks, attention, a scale router, and a regime router before producing one
residual. The MLP instead retains a direct path from flattened inputs.

This cycle asks:

> Does a zero-initialized direct shortcut from recent multiscale context to the
> persistence residual recover useful information that the deep custom path
> compresses away?

## Frozen architecture family

Every arm keeps the existing 3,611,530-parameter Fusion v2 deep branch.
Prediction remains:

\[
  \widehat y = y_{\mathrm{latest}} +
  \delta_{\mathrm{deep}} + \delta_{\mathrm{shortcut}}.
\]

The shortcut sees the Fourier horizon embedding plus only past context. It
never consumes target-time weather, calibration labels, or selection labels.
Its final projection is initialized to zero, so all arms begin with exactly the
same deep-branch prediction.

| candidate | shortcut input and function | parameters |
|---|---|---:|
| `shortcut-none` | zero contribution | 3,611,530 |
| `shortcut-history-linear` | all multiscale Cn2 level/difference/cadence values plus horizon; linear | 3,611,849 |
| `shortcut-all-linear` | history plus past operational-weather sequences plus horizon; linear | 3,612,941 |
| `shortcut-all-mlp` | same safe past inputs; LayerNorm--192 GELU--linear | 3,885,455 |

All models remain inside the immutable 2--8M parameter envelope. The shortcut
is an additive neural component of Strata-OT, not a LightGBM or external
prediction. Log the deep residual, shortcut residual, raw combined residual,
final residual, timescale weights, attention, expert weights, weather
attention, modulation, and physical-token norm.

Everything else is fixed: residual-horizon exponent zero, concatenative
multiscale weather interaction, full physical tokens, no pretraining, no
explicit point loss, raw Student-t scale, 12,000 training sequences, 4,000
selection examples, fold 1, seed 17, BF16, three epochs, batch size 128,
accumulation 2, learning rate `3e-4`, and weight decay `0.01`.

## Screen and decision

Rerun `shortcut-none` at the committed shortcut revision. An intervention
advances only if:

1. primary mean RMSE over 15/30/60 minutes improves by at least 2% over the
   paired no-shortcut control;
2. tail MAE and CRPS do not regress by more than 2%;
3. 80% interval coverage remains in 70--90%;
4. absolute bias is at most 0.25 log10 Cn2;
5. metrics are finite and provenance, artifacts, parameter envelope,
   component diagnostics, and resources pass.

The five-minute result remains an anchor. The lowest-RMSE
constraint-respecting arm advances to the unchanged three-fold, three-seed
protocol. If none advances, the three non-improvements close this direct
shortcut family; do not add another shortcut width or input combination after
viewing results.

## Leakage, safety, and budget

- Use only the verified public USNA-large snapshot and frozen purged rolling
  split. Never edit raw or sealed data.
- Fit preprocessing on fold training only, use calibration only for
  uncertainty records, and decide on selection only.
- Never load USNA confirmation labels or the official MLO test.
- Validate all three intervention proposal contracts before execution.
- One local training process, zero workers, four CPU threads, BF16, no cloud,
  and the existing RAM/VRAM/temperature/storage stop rules.
- Each arm is capped at 0.5 local GPU-hours; the cycle is capped at 2.0
  GPU-hours and remains inside the eight-hour program ceiling.
- No run begins after 2026-07-31 06:06 America/Chicago.

## Required evidence

Each run records the shortcut mode, parameter count, exact code/data/split
identities, checkpoint, predictions, deep/shortcut residual components,
calibration, runtime, board/process VRAM, RAM, storage, failures, and retry
state. The terminal result requires a frozen identity map, MLflow evidence
run, compiled and visually inspected LaTeX PDF, ledger entry, and verified
private Drive upload.
