# Frozen protocol: MLO weather-horizon ablation

Status: **frozen before implementation and assessment release**

Protocol parent commit:
`6400e060d2204ce0350b5da7e31b80c4646b6aff`

Planned experiment identity: `mlo-weather-horizon-v1`

This is a development experiment. It cannot promote a champion, weaken an
evaluation gate, or justify a public performance claim on the official MLO
test partition.

## Plain-language question

Recent turbulence measurements are usually a strong short-term forecast:
copying the latest measurement is hard to beat. This experiment asks whether
ordinary operational weather measurements add useful information as the
forecast moves from 5 minutes to 1 hour, and whether a purpose-built model can
use that information more effectively than a plain neural network or
LightGBM.

The experiment deliberately uses only six recent observations and six
operational weather variables. It does not give the model turbulence-derived
flux covariances, quality flags, sample counts, or instrument diagnostics,
because those fields may be target proxies rather than information available
to an honest operational forecast.

## Falsifiable hypothesis

For the average of the 15-, 30-, and 60-minute horizons, Strata-OT Horizon v1
with operational weather will reduce assessment RMSE by at least 2% relative
to both:

1. the same Horizon v1 architecture with its weather contribution disabled;
2. the stronger operational-weather control between the compact MLP and
   LightGBM.

The improvement must survive the preregistered paired block bootstrap, have
the same direction for both neural seeds, retain calibrated uncertainty, and
avoid material regressions in bias, tail error, CRPS, or seed stability. The
5-minute result is a reported anchor and is not part of the primary endpoint.

The null result is that operational weather does not meet all of those
conditions. A partial or null result will be reported without architecture
retuning after assessment release.

## Fixed evidence boundary

- Source dataset: `otbench-mlo-cn2-15m-v1`
- Pinned OTBench commit:
  `53cab9d53648b4870b43e35d48f19143786fa12e`
- Checked-in dataset manifest:
  `data/manifests/otbench-mlo-cn2-15m-v1.json`
- Manifest SHA-256:
  `cd1bba8a0ca6ddc2a25f5d40a8f7b6d0187b48ad2c119d03d917c869980cf410`
- Existing outer split:
  `otbench-mlo-blocked-v1`
- Existing outer split SHA-256:
  `200c09ae94e1f946e7d079d89636854fb97531d99c4e6a79a7d9ed49d1b93b85`
- Geometry: one 15 m surface point at Mauna Loa Observatory
- Label: direct observed `log10(Cn2_15m)` in
  `log10(m^(-2/3))`

No new dataset will be acquired. Raw data will never be edited. The official
MLO test partition must not be opened by this experiment, even though the
prior acquisition snapshot contains sealed test files. The previously
released USNA confirmation is context only and must not be used for tuning,
selection, calibration, or success decisions.

## Frozen development split

The only evaluation source is the official 1,892-row MLO validation partition.
Its `_upstream_index` timestamp is the row identity.

1. Preserve the existing 24-row purge at both outer validation boundaries.
   This removes raw validation offsets `0..23` and `1868..1891`, leaving 1,844
   rows at offsets `24..1867`.
2. Divide those 1,844 rows chronologically into nominal blocks of 922, 461,
   and 461 rows (50%, 25%, and 25%).
3. At each internal boundary, remove 24 rows from the earlier side and 24 rows
   from the later side.

The resulting role boundaries are fixed:

| Role | Raw validation offsets | First identity | Last identity | Rows |
| --- | ---: | --- | --- | ---: |
| Selection | `24..921` | `2006-07-10 03:27:30` | `2006-07-13 09:47:30` | 898 |
| Calibration | `970..1382` | `2006-07-13 14:27:30` | `2006-07-16 03:12:30` | 413 |
| Assessment | `1431..1867` | `2006-07-16 07:27:30` | `2006-07-17 21:57:30` | 437 |

Offsets `922..969` and `1383..1430` are purged. Exact ordered identities for
all three roles must be materialized under ignored `data/sealed/` storage
before training. Only their role counts, endpoint identities, and SHA-256
hashes may be checked in. A verification command must fail on count, order,
endpoint, disjointness, or hash drift.

Training rows come only from the official training partition after the
existing train/validation boundary purge. Median imputation, feature
normalization, and target normalization are fit only on that training block.
Selection labels may be used only for early stopping and fixed comparator
diagnostics. Calibration labels may be used only to fit the single scalar
uncertainty multiplier after point-model training. Assessment labels may be
loaded only when `--release-assessment` is explicitly supplied.

Before assessment release, the validation target reader must stop at raw
offset 1382; it may not read and then discard assessment rows. The new
development release flag is independent of the existing one-shot
`--release-test` interface for USNA. No code path in this experiment may load
`test_features.csv` or `test_target.csv`.

## Forecast examples and cadence

- Context: six rows, representing approximately 30 minutes.
- Horizon rows: `1`, `3`, `6`, and `12`.
- Horizon minutes: `5`, `15`, `30`, and `60`.
- Nominal cadence: 5 minutes.
- Every timestamp must be strictly increasing.
- Each context gap must be no more than 7.5 minutes.
- Target elapsed time from the final context row must equal the nominal
  horizon within 2.5 minutes. Examples crossing a larger missing-data gap are
  rejected rather than silently changing the forecast horizon.

All learned models are trained on the same stacked four-horizon examples.
They receive an explicit horizon identity and are evaluated separately at
every horizon. This makes the horizon-conditioned model meaningful while
holding its training rows fixed. The deterministic baselines are computed
once per horizon. MLflow tags must include the ordered row and minute
identities (`1,3,6,12` and `5,15,30,60`) and the feature-set identity.

## Frozen feature sets

### `history_only`

- six historical `log10(Cn2_15m)` levels;
- first differences derived from those levels inside Horizon v1;
- elapsed cadence ratios relative to 5 minutes.

The controls receive the same history, cadence, and numeric horizon identity.
No current or future target is present in the inputs.

### `operational_weather`

The history-only inputs plus the following six raw allowlisted fields at each
context row:

- `Dir_10m`
- `Spd_10m`
- `P_2m`
- `T_2m`
- `RH_2m`
- `Tdew_2m`

`Dir_10m` is replaced by sine and cosine channels before normalization, so the
weather level branch has seven numeric channels. Weather first differences
are taken after circular encoding. Raw direction degrees are not supplied as
an additional channel.

The raw feature allowlist must be explicit in configuration and the resolved
allowlist must be logged with every learned run.

### Excluded primary fields and proxy audit

The primary experiment excludes every MLO covariance/flux, QC flag, count,
and instrument-diagnostic field, including names matching covariance pairs,
`*flag*`, `diag*`, `count*`, and water-vapor instrument channels. It also
excludes non-allowlisted wind and sonic-temperature fields.

Before assessment release, a training-only feature audit will list all 87
source fields by category, missingness, and association with the label.
Potential target proxies will be identified from provenance and
training-only association statistics. The audit is descriptive: no excluded
field may be added to a learned arm, and there is no promotion-eligible
"all-fields" experiment.

## Comparator matrix

For every horizon:

- climatology: official-training target mean;
- persistence: latest valid Cn2 history level;
- recent mean: mean of the six Cn2 history levels.

For both feature sets:

- LightGBM, one deterministic stacked-horizon run;
- compact MLP, seeds 17 and 41;
- existing Strata-OT Surface, seeds 17 and 41;
- Strata-OT Horizon v1, seeds 17 and 41.

All learned controls receive the same feature arm and scalar horizon identity.
Metrics and predictions are disaggregated by horizon even when one fitted
model serves all four horizons.

## Strata-OT Horizon v1

Horizon v1 is a 2--5 million parameter custom adaptation, not a universal
novelty claim.

1. **History representation.** Build per-row tokens from the Cn2 level, first
   difference, and cadence ratio, then project to width 192.
2. **Weather representation.** Build separate tokens from seven weather
   levels and their first differences, then project to width 192.
3. **Causal encoders.** Each branch uses width-192 gated causal temporal
   blocks with kernel size 3 and dilations 1 and 2. Padding must not expose a
   later context token to an earlier token.
4. **Horizon query.** Encode forecast minutes with fixed Fourier sine/cosine
   features and a learned width-192 projection.
5. **Dual attention.** Use separate six-head horizon-to-history and
   horizon-to-weather cross-attention modules. The horizon is the query and
   the branch tokens are keys and values.
6. **Regime routing.** Fuse horizon, history, and weather summaries through
   the existing four-expert Strata regime router.
7. **Interpretable residual.** The point location is

   `persistence + history_delta + sigmoid(weather_gate) * weather_delta`.

   The three scalar components and gate are logged. In `history_only`,
   `weather_delta` and the applied weather contribution are hard-zeroed; no
   missing-weather embedding can leak a learned weather term.
8. **Distribution head.** Retain the existing Gaussian scale and
   0.10/0.50/0.90 quantile machinery. The explicit residual supplies the
   location; learned quantile offsets and the positive scale come from the
   same fused state. No new probability family is introduced.
9. **Diagnostics.** Log regime weights, weather gates, history and weather
   deltas, applied weather contribution, both attention summaries, and their
   per-horizon distributions.

The implementation must assert a parameter count from 2,000,000 through
5,000,000 inclusive. Literature reviewed before this freeze found close
components in TimeXer, MQTransformer, TFT, TiDE, PatchTST, CAMul, and FreqMoE,
but not this exact combination. The architecture will therefore be described
as a custom adaptation. Literature findings cannot trigger a post-freeze
redesign.

## Fixed training and calibration policy

- Framework: PyTorch Lightning through the repository Prefect flow.
- Precision: BF16 mixed precision on the RTX 5080.
- Maximum epochs: 80.
- Early-stopping patience: 12 epochs.
- Early-stopping data: selection only.
- Learning rate: `3e-4`.
- Weight decay: `0.01`.
- Seeds: 17 and 41 for every neural family and feature arm.
- Workers: `STRATA_NUM_WORKERS=0`.
- One training process and one GPU only.
- CPU threads: at most four for PyTorch, OpenMP, MKL, and OpenBLAS.
- Calibration: one positive scalar multiplier per neural run, fit on pooled
  calibration examples only and then reported separately by horizon.
- OOM recovery: at most once, by halving batch size and doubling gradient
  accumulation. Model width, data, horizons, and evidence rules remain fixed.

No optimizer, feature, architecture, epoch, calibration, or success-rule
change is allowed after assessment release.

## Metrics and primary endpoint

Report at every horizon:

- RMSE, MAE, and bias in log10 Cn2;
- Gaussian CRPS and NLL;
- nominal 80% interval coverage and width;
- top-decile target MAE;
- correlation when numerically defined;
- inference latency, runtime, peak process RAM, peak allocated/reserved VRAM,
  storage delta, and local GPU-hours;
- Horizon v1 component and routing diagnostics.

For model `m`, define its primary RMSE as the arithmetic mean of its
horizon-specific RMSE values at 15, 30, and 60 minutes. Define relative
improvement against comparator `c` as the arithmetic mean across those
horizons of `1 - RMSE(m,h) / RMSE(c,h)`. Neural headline values average the
two seed-specific metrics; individual seeds remain visible.

The stronger operational-weather control is the smaller assessment primary
RMSE of operational-weather MLP and operational-weather LightGBM. Horizon v1
must beat both controls in the result table; describing the stronger one is
only a concise naming convention, not a comparator-selection tuning step.

## Frozen success rule

Operational-weather Horizon v1 succeeds only if all conditions hold:

1. Mean relative RMSE improvement over 15/30/60 minutes is at least 2.00%
   against both history-only Horizon v1 and the stronger operational-weather
   MLP/LightGBM control.
2. A paired circular moving-block bootstrap using nominal 24-hour blocks (288
   five-minute rows), 2,000 resamples, and seed `20260730` has a 95% interval
   strictly above zero for each of those two improvements. Each resample
   recomputes per-horizon RMSE, the three-horizon relative improvement, and
   then the mean across neural seeds. Deterministic controls are repeated
   across seeds.
3. Seeds 17 and 41 each have positive three-horizon improvement against both
   comparators.
4. Each operational-weather Horizon v1 seed at each primary horizon has 80%
   interval coverage from 70% through 90%, inclusive.
5. Absolute bias is at most the immutable 0.25 log10 gate and no more than
   0.02 log10 worse than the stronger control.
6. Three-horizon tail MAE and CRPS are each no more than 2% worse than the
   stronger control.
7. The relative standard deviation of the two seed primary RMSE values is at
   most 0.15 and no more than 0.02 above the corresponding history-only
   Horizon v1 stability value.
8. Provenance, split, preprocessing, numerical, storage, hardware, report,
   and reproducibility gates all pass.

The 5-minute anchor, parameter count, weather gate, regime weights, and
component deltas are reported regardless of outcome but cannot rescue a
failed primary rule. No model is promoted automatically.

## Assessment-release sequence

1. Verify the source manifest and materialize/hash role identities without
   loading assessment labels.
2. Commit this protocol.
3. Implement the split, features, models, orchestration, logging, API,
   console, proposal, tests, and report source.
4. Run tests, manifest/split verification, proposal validation, and
   selection/calibration-only training or dry runs as needed.
5. Commit the complete implementation and resolved frozen configuration.
6. Confirm the committed worktree is clean and record the commit in the run
   manifest.
7. Invoke the bounded Prefect experiment once with
   `--release-assessment`.
8. Do not retune. Evaluate all gates and report positive, partial, null, or
   failed evidence exactly.

## Logging and artifact contract

Every MLflow run and failure must record:

- repository commit and dirty state;
- dataset, manifest, split, role, feature-set, and horizon identities/hashes;
- exact raw allowlist and resolved tensor channels;
- resolved model, trainer, optimizer, seed, batch, and recovery state;
- environment and package fingerprint;
- CPU/RAM/GPU/VRAM identity and time series where available;
- runtime, storage delta, local GPU-hours, and cloud cost (`$0`);
- checkpoints, predictions, per-horizon metrics and plots;
- calibration fit record;
- Horizon v1 components, gates, attention, and regime diagnostics;
- failure, retry, and stop reason.

The API and research console must expose the real feature-by-horizon matrix,
actual gate outcomes, run/artifact links, Horizon v1 component summaries, and
a plain-language conclusion. Demo or fabricated values are prohibited.

## Hardware, budget, storage, and deadline

- Combined new neural training: less than 4 GPU-hours.
- Cloud spending: exactly `$0`.
- New overnight storage: at most 25 GiB.
- Preserve at least 50 GiB free on `C:`.
- No dataset download is planned. A required public download above 5 GiB
  would go to `I:\StrataData\<dataset-id>` only after provenance and size
  checks.
- Never delete unrelated data, prune Docker globally, or interfere with
  voxel-game processes.

Do not start a run if CPU exceeds 85% for three sustained minutes, free RAM is
below 12 GiB, free GPU memory is below 4 GiB, or GPU temperature exceeds
75 C. Gracefully checkpoint and stop if free RAM falls below 8 GiB, total
board VRAM exceeds 15 GiB, process peak VRAM exceeds 14 GiB, or GPU
temperature exceeds 84 C for 60 seconds.

At **2026-07-30 12:30 PM America/Chicago**, start no new run, download,
architecture revision, code revision, or research cycle. A bounded run
already in flight may finish. After that gate, only evidence finalization,
report compilation/inspection, Drive upload, and handoff are allowed.

Stop early on provenance or leakage failure, corrupt labels, manifest drift,
repeated numerical failure, one failed OOM recovery, unsafe
RAM/VRAM/temperature/storage, cloud or gated-data requirements, inability to
preserve reproducible evidence, or any request for champion promotion or
public release.

## Required reports and delivery

The architecture/literature review and final experiment report must be
generated from checked-in LaTeX, compiled to PDF, rendered, and visually
inspected. The final report must include the frozen plan, beginner-readable
summary, provenance, feature-proxy audit, methods, closest prior art, exact
configuration, all horizon results, uncertainty, failures, limitations,
novelty assessment, claim-to-evidence mapping, resource/storage use, and next
recommendation.

After validation, the branch is pushed and a draft PR is opened. It is not
merged. PDFs, the earlier first-real-run PDF, and a compact run index are
uploaded to the requested Google Drive folder without raw data, sealed
identities, checkpoints, MLflow internals, credentials, or sharing changes.

