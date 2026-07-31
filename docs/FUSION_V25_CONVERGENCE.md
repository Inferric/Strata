# Fusion v2.5 fixed-architecture convergence cycle

Status: **frozen 2026-07-30 23:25 America/Chicago; no convergence arm has run**

Parent: `docs/FUSION_V24_DIRECT_SHORTCUT.md`

Parent evidence run: `0e8b0f954fa14af9a714cd67adbecbd7`

Evaluation role: rolling development selection only

## Evidence-motivated question

The fixed history-linear shortcut improved mean 15/30/60-minute RMSE by
3.05% over the paired no-shortcut model after three epochs, but it failed the
tail and 70--90% coverage constraints. Its 319-parameter shortcut learned a
non-trivial residual while the deep branch also moved substantially. This
cycle asks:

> Does allowing that exact architecture to converge for 6, 9, or 12 epochs
> retain at least a 2% point-forecast gain while restoring tail error and
> interval coverage?

This is an optimization-duration family, not another shortcut search. The
shortcut input, width, initialization, probability head, data, and all gates
are fixed. No additional shortcut input, width, gate, loss, or calibration
variant may be introduced after results are viewed.

## Frozen arms

Every arm is the 3,611,849-parameter `history_linear` architecture from v2.4:

\[
  \widehat y = y_{\mathrm{latest}} +
  \delta_{\mathrm{deep}} + \delta_{\mathrm{history\ shortcut}}.
\]

| candidate | maximum fine-tuning epochs |
|---|---:|
| `convergence-3` | 3 |
| `convergence-6` | 6 |
| `convergence-9` | 9 |
| `convergence-12` | 12 |

Early stopping continues to monitor rolling fold-1 selection RMSE with the
unchanged patience. Longer caps do not authorize selection-label calibration
or checkpoint choice on another role. The no-shortcut baseline is the frozen
clean run `46af532e62ed4aaf9f50b9faab1e131d` from the same preceding architecture
cycle. `convergence-3` is rerun at the committed convergence revision to
separate code identity from the historical comparison.

All other settings remain fixed: history-linear shortcut, residual-horizon
exponent zero, concatenative multiscale weather interaction, full physical
tokens, no pretraining, no explicit point loss, raw Student-t scale, 12,000
training sequences, 4,000 calibration examples, 4,000 selection examples,
fold 1, seed 17, BF16, batch size 128, accumulation 2, learning rate `3e-4`,
and weight decay `0.01`.

## Frozen decision

An intervention advances only if, relative to the frozen no-shortcut baseline:

1. primary mean RMSE over 15/30/60 minutes improves by at least 2%;
2. tail MAE and CRPS do not regress by more than 2%;
3. 80% interval coverage remains in 70--90%;
4. absolute bias is at most 0.25 log10 Cn2;
5. metrics are finite and provenance, artifacts, parameter identity,
   convergence records, component diagnostics, and resources pass.

The five-minute result remains an anchor. Among advancing arms, select the
lowest primary RMSE; a tie within 0.25% selects fewer epochs. If none advances,
the three longer-duration non-improvements close this convergence family.
There is no duration interpolation after metrics.

## Leakage, safety, and budget

- Use only the verified public USNA-large snapshot and frozen purged rolling
  fold. Never edit raw or sealed data.
- Fit preprocessing on fold training only, use calibration only for
  uncertainty records, and decide on selection only.
- Never load USNA confirmation labels or the official MLO test.
- Validate all three intervention proposal contracts before execution.
- One local training process, zero workers, four CPU threads, BF16, no cloud,
  and the existing RAM/VRAM/temperature/storage stop rules.
- Each arm is capped at 0.5 local GPU-hours; the cycle is capped at 2.0
  GPU-hours and remains within the eight-hour program ceiling.
- No run begins after 2026-07-31 06:06 America/Chicago.

## Required evidence

Each run records requested and completed epochs, optimizer steps, exact
code/data/split identities, checkpoint, predictions, deep/shortcut residual
components, calibration, runtime, board/process VRAM, RAM, storage, failures,
and retry state. The terminal result requires a frozen identity map, MLflow
evidence run, compiled and visually inspected LaTeX PDF, ledger entry, and
verified private Drive upload.
