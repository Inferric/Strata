# Fusion v2.1 explicit point-loss cycle

Status: **frozen 2026-07-30 22:18 America/Chicago; no point-loss arm has run**

Parent: `docs/FUSION_V2_ROBUSTNESS.md`

Parent evidence run: `5b2f104cb416450eb82b7ef957df10cf`

Evaluation role: rolling development selection only

## Frozen parent evidence

The complete 45-run clean robustness matrix was assessed before this cycle was
frozen. Fusion v2 achieved 0.255197 primary RMSE and improved 4.07% over
persistence, with positive seed direction for seeds 17, 41, and 73. It was
1.26% worse than the stronger MLP control; the paired 24-hour bootstrap
interval versus MLP was entirely negative, and tail MAE materially regressed.
The immutable robustness result is therefore `FAIL`. Coverage, CRPS, bias, and
stability passed. These facts motivate the single objective-level question
below but do not alter its weights, gates, or denominator.

## Evidence-motivated question

The frozen Fusion v2 candidate predicts only a small correction to persistence,
and several full runs selected very early checkpoints. Its training objective
is Student-t negative log likelihood plus a 0.20-weight quantile loss. A
heavy-tail scale can reduce likelihood loss without requiring the location
head to learn a useful persistence residual.

This cycle asks one falsifiable question:

> Does an explicit robust loss on the forecast location make Fusion v2 learn a
> useful persistence residual without changing its architecture, probability
> distribution, data, or evidence thresholds?

This is a training-objective hypothesis. It does not reinterpret the completed
robustness matrix and cannot open confirmation evidence.

## Frozen intervention

Keep the selected Fusion v2 architecture unchanged:

- 192 hidden units, six heads, four diagnostic experts, and about 3.61M
  parameters;
- short, medium, and slow Cn²/weather histories;
- concatenative weather interaction at every temporal scale;
- persistence-anchored residual and Fourier horizon decoder;
- full physical/refractivity tokens;
- no pretraining initialization;
- raw learned Student-t scale and the existing ordered quantiles.

The existing objective is

`Student-t NLL + 0.20 × pinball loss`.

Add PyTorch Smooth L1 loss between the location forecast and target, with
`beta = 0.10`, at exactly three intervention weights. Rerun a zero-weight
control at the same code revision so probability calibration and sample caps
are exactly paired:

| candidate | point-loss weight |
|---|---:|
| `point-huber-0p00` | 0.00 (control) |
| `point-huber-0p25` | 0.25 |
| `point-huber-0p50` | 0.50 |
| `point-huber-1p00` | 1.00 |

No other factor may change inside this family.

## Screen and decision

Run all four arms on fold 1, seed 17, BF16, zero workers, three epochs maximum,
up to 12,000 chronological training examples, and up to 4,000 selection
examples through Prefect and MLflow. Compare the three interventions against
the new zero-weight control. The earlier frozen `fusion-concat` screen remains
context, not the paired denominator.

An arm advances only if all of the following hold:

1. primary mean RMSE over 15/30/60 minutes improves by at least 2% over the
   paired zero-weight control;
2. tail MAE and CRPS do not regress by more than 2%;
3. 80% interval coverage remains in 70--90%;
4. bias, numerical stability, provenance, artifacts, and resources pass.

If none advances, the three non-improvements close the explicit point-loss
family. Do not add another weight after seeing results. Pivot to the already
motivated data-scale hypothesis instead of retuning this loss.

If one arm advances, freeze the lowest-RMSE constraint-respecting arm and run
it on all three rolling folds and seeds 17, 41, and 73. Full-run success still
requires the unchanged 2% comparisons, 24-hour paired bootstrap, three-seed
direction, coverage, tail, CRPS, bias, and stability gates from the parent
protocol.

## Leakage, safety, and budget

- Fit preprocessing on fold training only, early-stop on selection only, and
  use fold calibration only for uncertainty records.
- Never load USNA confirmation labels or the official MLO test.
- Validate all three intervention proposal contracts before execution; the
  paired zero-weight arm is an unchanged control, not an autonomous proposal.
- One local training process, zero workers, four CPU threads, BF16, no cloud,
  and the existing RAM/VRAM/temperature/storage stop rules.
- Each arm is capped at 0.5 local GPU-hours; the four-run cycle is capped at
  2.0 GPU-hours and remains inside the eight-hour program ceiling.
- No run begins after 2026-07-31 06:06 America/Chicago.

## Required evidence

Each run records the exact code/data/split identities, objective weight,
checkpoint, predictions, component diagnostics, calibration, runtime,
board/process VRAM, RAM, storage, failures, and retry state in MLflow. The
cycle ends with terminal gate states, a ledger event, and a compiled,
visually inspected LaTeX report or is incorporated explicitly into the final
program synthesis.
