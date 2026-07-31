# Fusion v2.2 training-sequence-density cycle

Status: **frozen 2026-07-30 22:32 America/Chicago; no data-scale arm has run**

Parent: `docs/FUSION_V21_POINT_LOSS.md`

Parent evidence run: `359831309fba4724be654f38ba81f7cc`

Evaluation role: rolling development selection only

## Evidence-motivated question

The complete Fusion v2 robustness matrix improved over persistence but not the
stronger MLP control and materially regressed tail MAE. A paired explicit
point-loss screen then produced only 0.50--0.67% primary RMSE improvement, well
below the immutable 2% advancement threshold; two larger weights also regressed
tail MAE. That closes the explicit point-loss family without changing the
original objective.

The original selection screens used only 12,000 evenly spaced training
sequences. Fold 1 contains exactly 62,263 valid sequences under the frozen
adapter. This cycle asks:

> Does denser use of the same chronological public training fold improve the
> unchanged Fusion v2 model enough to pass the existing error, tail,
> calibration, provenance, and resource gates?

This is a sample-density hypothesis. It does not add dates, alter a boundary,
or provide evidence about broader seasonal coverage.

## Frozen intervention

Keep the selected Fusion v2 candidate unchanged:

- 192 hidden units, six attention heads, four diagnostic experts, and 3,611,530
  parameters;
- short, medium, and slow Cn2/weather histories;
- concatenative weather interaction, full physical/refractivity tokens, and a
  persistence-anchored residual;
- Student-t negative log likelihood plus the existing 0.20-weight quantile
  loss, with explicit point-loss weight fixed at zero;
- raw learned Student-t scale, ordered quantiles, no pretraining initialization,
  BF16, batch size 128, accumulation 2, learning rate `3e-4`, and weight decay
  `0.01`.

Only the maximum number of training sequences changes:

| candidate | maximum / actual training sequences |
|---|---:|
| `data-scale-12k` | 12,000 (paired control) |
| `data-scale-30k` | 30,000 |
| `data-scale-50k` | 50,000 |
| `data-scale-full` | 62,263 (all valid fold-1 sequences) |

The dataset adapter deterministically chooses evenly spaced indices from the
same chronological sequence list when a cap applies. All arms therefore span
the same fold-1 dates; larger arms make that coverage denser. Training-only
normalization is fit once from the complete fold-1 training role for every arm,
so preprocessing is identical and never sees calibration or selection labels.

## Screen and decision

Run all four arms on fold 1, seed 17, BF16, zero workers, three epochs maximum,
and the same 4,000 selection examples through Prefect and MLflow. Rerun the
12,000-sequence control at the committed data-scale revision so all four arms
have identical code and evidence identities.

An intervention advances only if all of the following hold:

1. primary mean RMSE over 15/30/60 minutes improves by at least 2% over
   `data-scale-12k`;
2. tail MAE and CRPS do not regress by more than 2%;
3. 80% interval coverage remains in 70--90%;
4. absolute bias is at most 0.25 log10 Cn2;
5. metrics are finite and provenance, artifacts, parameter identity, and
   resource envelopes pass.

The five-minute horizon remains a reported anchor, not part of the primary
average. The lowest-RMSE constraint-respecting arm advances. If none advances,
the three non-improvements close this within-fold sample-density family; do not
add an intermediate cap after viewing results. Pivot to a separately
preregistered representation or training-target hypothesis.

If one advances, freeze its exact sequence cap and run the unchanged full
three-fold, three-seed protocol. It must still pass the parent 2% control
comparisons, paired 24-hour block bootstrap, seed-direction, coverage, bias,
tail, CRPS, stability, provenance, and resource gates before confirmation can
even be considered.

## Leakage, safety, and budget

- Use only the already verified public USNA-large snapshot and frozen purged
  rolling split. Never edit raw or sealed data.
- Fit preprocessing on fold training only, train on training sequences only,
  use calibration only for uncertainty records, and decide on selection only.
- Never load USNA confirmation labels or the official MLO test.
- Validate all three intervention proposal contracts before execution; the
  12,000-sequence arm is an unchanged paired control.
- One local training process, zero workers, four CPU threads, BF16, no cloud,
  and the existing RAM/VRAM/temperature/storage stop rules.
- Each arm is capped at 0.5 local GPU-hours; the four-run cycle is capped at
  2.0 GPU-hours and remains inside the eight-hour program ceiling.
- No run begins after 2026-07-31 06:06 America/Chicago.

## Required evidence

Each run records the requested cap, actual training-sequence count, exact
code/data/split identities, checkpoint, predictions, components, calibration,
runtime, board/process VRAM, RAM, storage, failures, and retry state. The cycle
ends with immutable terminal gates, a chronological ledger event, a compiled
and visually inspected LaTeX report, an MLflow evidence run, and a verified
private Drive upload.
