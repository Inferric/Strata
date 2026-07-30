# Strata-OT Fusion v2 autonomous research program

Status: preregistered; confirmation labels not released  
Program ID: `strata-fusion-v2-program`  
Start: 2026-07-30 18:06 America/Chicago  
No-new-run deadline: 2026-07-31 06:06 America/Chicago

## Plain-language question

Cycle 0 showed that ordinary weather measurements contain useful information:
LightGBM used them, but Strata-OT Horizon v1 usually did not. This program asks
whether a neural model can learn those interactions more reliably when it sees
weather throughout its temporal representation, sees several history
timescales, and first learns auxiliary atmospheric tasks. The program is
development research, not a champion-promotion exercise.

## Prior evidence and scope

- The completed MLO weather-horizon experiment is Cycle 0. Its released
  assessment is consumed and may be cited but never used for tuning.
- The official MLO test partition remains sealed and must never be loaded.
- The public OTBench USNA-large NetCDF at upstream commit
  `53cab9d53648b4870b43e35d48f19143786fa12e` is the only data source.
- The source SHA-256 is
  `c38a974d71ba58b9029687ad3397d6c1330c16c53e8a51ad1a94a3fcec3dbd4f`.
  Raw bytes are immutable and stay outside Git.
- An audit read aggregate USNA-large validation target values before this
  protocol was frozen. Validation is therefore ineligible as untouched
  confirmation evidence even though no model was tuned on those aggregates.
- Development uses only the official 2020 training partition. A single
  preregistered March--April 2022 slice of the public official USNA-large test
  partition is reserved for the final frozen candidate.

## Label, covariates, and sampling

The direct target is `Cn2_3m` in \(m^{-2/3}\), modelled as
\(\log_{10}(C_n^2)\). Available public covariates are `T_3m`, `P_3m`,
`RH_3m`, `Spd_3m`, `Rad_1m`, `T_0m`, `Dir_3m`, and time. The primary cadence
is a deterministic five-minute endpoint grid derived from the one-minute
source; no random row split or interpolation across a rejected gap is allowed.

Three nested context branches end at the same forecast origin:

| Branch | Samples | Spacing | Approximate span |
| --- | ---: | ---: | ---: |
| short | 6 | 5 minutes | 30 minutes |
| medium | 12 | 15 minutes | 3 hours |
| slow | 24 | 60 minutes | 24 hours |

Forecast horizons are fixed at 5, 15, 30, and 60 minutes. Histories with a
source gap that makes a requested lag or target unavailable are rejected.
Missing covariates carry explicit masks. Wind direction is encoded as sine and
cosine only where present; its high missingness prevents reliance on it.

Derived physics features use only contemporaneous allowlisted weather:
dew point, vapour pressure, dry and wet refractivity terms, total refractivity,
air--surface temperature difference, pressure and temperature tendencies,
wind components, radiation, and cyclic time features. They are deterministic
covariates, not replacement labels or causal regime annotations.

## Frozen rolling-origin development split

All offsets below refer to the pinned `usna_cn2_lg.nc`. Role identities are
materialized locally and hashed before model fitting. Each role begins at least
2,880 minutes after the preceding role, which exceeds the 24-hour maximum
context plus the 60-minute horizon. Preprocessing is fitted on that fold's
training role only. Early stopping and architectural selection use selection
only; scalar/quantile calibration uses calibration only.

| Fold | Role | UTC period (inclusive) | Raw offsets | Rows | Identity SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| 1 | train | 2020-01-01 00:00--2020-03-15 23:59 | 0--106630 | 106631 | `c4eaa519d28f0af9e06eb150bfeb87c1a9c428c6be52d7f66646b32bdd449180` |
| 1 | calibration | 2020-03-18 00:00--2020-03-24 23:59 | 109511--119590 | 10080 | `263f2565da3dc20d4e0efe9c3308aa222c5f2400e6650a1d00f3af10d23ac619` |
| 1 | selection | 2020-04-01 00:00--2020-04-30 23:59 | 129670--172843 | 43174 | `db0dfa5bd075469463aca03411473a6971d7a319ae5090ab1a59f64c61dfa6a3` |
| 2 | train | 2020-01-01 00:00--2020-06-15 23:59 | 0--239081 | 239082 | `c608e067827b80fa9b7fde2f4bc567241e991ee16ca21b6b8c78ea5f30296fe8` |
| 2 | calibration | 2020-06-18 00:00--2020-06-24 23:59 | 241962--252041 | 10080 | `eb73526fb0f5ead583c5f8ae0b1162c7de7f0746b5b81b2e516987d91c673f35` |
| 2 | selection | 2020-07-01 00:00--2020-07-31 23:59 | 260682--305320 | 44639 | `070c90fa64636edf1a06f0d1c07fdd53900feeea7da991aff2e39752616778ed` |
| 3 | train | 2020-01-01 00:00--2020-09-15 23:59 | 0--371560 | 371561 | `b03a908772dc87b7c53fb5a8662d4d53637f8190a4b4cb99d7d58f7fdc5cfc5e` |
| 3 | calibration | 2020-09-18 00:00--2020-09-24 23:59 | 374441--384520 | 10080 | `cd15c50035a0e5cba83390bc041142c68e5117b9cac46283c5d2dc4f28b6e1ed` |
| 3 | selection | 2020-10-01 00:00--2020-10-31 23:59 | 393161--437557 | 44397 | `484c33e77970cb4af7af45fead5a2c73f06330032517ca241c2cac03ee71210c` |
| final | train | 2020-01-01 00:00--2020-12-15 23:59 | 0--502071 | 502072 | `e9064e26429b2231821b1fa2611998866556aa761ddd0b58384b2c3d954caeb5` |
| final | calibration | 2020-12-18 00:00--2020-12-24 23:59 | 504952--514910 | 9959 | `d78ed40534cb600a18aca19999b0d39827dd21d2751a97f839554fcd260ab1ff` |

## One-shot confirmation

Confirmation is the inclusive UTC interval 2022-03-01 00:00 through
2022-04-30 23:59, raw offsets 1,096,892--1,184,664 (87,773 identities),
identity SHA-256
`23c127fb4d7a46f730dfcff497645918f19bb1ba4115cf6d1ef2442d46e111e7`.
It is within the upstream official USNA-large test partition. The adapter must
refuse to load its target unless all of the following are true:

1. protocol, implementation, candidate identity, and calibration policy have
   committed Git identities;
2. the candidate was selected without confirmation labels;
3. the command includes `--release-confirmation`;
4. the confirmation is not already in the consumption registry.

Release consumes the slice permanently. No result-driven model, threshold,
calibration, or report-claim change may follow. All remaining official
USNA-large test rows stay unreleased.

## Strata-OT Fusion v2

The custom target is 2--8 million trainable parameters:

- causal encoders for short, medium, and slow \(C_n^2\) histories;
- persistence-anchored residual prediction for all four horizons;
- horizon Fourier queries and horizon-conditioned multi-scale pooling;
- a weather encoder that modulates temporal blocks using feature-wise affine
  modulation and cross-attention, not a suppressible final scalar gate;
- the deterministic physics tokens and missingness masks above;
- four learned representation experts with descriptive mixture weights (no
  causal regime claim);
- monotone 10/50/90% quantiles plus a Student-t location, scale, and degrees of
  freedom head;
- OOD distance from training embeddings and selective-risk curves;
- logged scale weights, weather attention, modulation norms, physics-token
  norms, residual components, and expert weights.

The prediction median is
`last_observation + horizon_conditioned_residual`. All uncertainty and OOD
fitting uses fold training/calibration roles only.

## Training-only atmospheric pretraining

Pretraining never reads selection, confirmation, or MLO assessment/test labels.
On each fold's training role it combines:

- masked reconstruction of observed weather values;
- multi-horizon future-weather prediction;
- cadence-aware future \(C_n^2\) prediction;
- deterministic physics-token reconstruction where inputs are observed.

Loss weights, masks, and pretraining epochs are part of the candidate config.
Fine-tuning reuses the encoder and records whether it was initialized from the
fold-local checkpoint.

## Frozen ablation program

Cheap screens use fold 1, seed 17, capped data, and selection-only metrics.
Every category is executed even if it is negative:

1. context: short only; short+medium; all three scales;
2. fusion: concatenation; FiLM; FiLM plus cross-attention;
3. physics: absent; base refractivity/stability tokens; full tokens;
4. pretraining: none; reconstruction only; full multitask;
5. capacity/regularization: approximately 2M, 4M, and 7M with fixed dropout
   candidates;
6. probability: raw quantiles, calibrated quantiles, and Student-t plus
   calibrated quantiles.

Controls are climatology, persistence, MLP, TCN, diagnostic LightGBM, Horizon
v1, and the best prior Fusion v2 candidate. LightGBM feature importance and
training-fold permutation analyses may motivate neural features, but the tree
model is neither copied into nor shipped as Strata-OT.

A candidate advances only when its fold-1 mean RMSE across 15/30/60 minutes is
better than the current custom candidate, it is numerically stable, and it does
not worsen 80% coverage distance from 80% or tail MAE by more than 2%. Ties
within 0.25% prefer the smaller model. The best surviving architecture is run
on all three rolling folds and seeds 17, 41, and 73. Three consecutive
non-improvements close a hypothesis family and force the next preregistered
category rather than ending the program.

## Metrics and immutable evidence rules

The 5-minute horizon is an anchor. The primary development and confirmation
endpoint is mean relative RMSE improvement over 15/30/60 minutes. Report per
horizon RMSE, MAE, bias, 95th-percentile absolute error, CRPS, 80% interval
coverage/width, Student-t NLL, seed/fold stability, OOD-stratified error,
selective risk, latency, total board VRAM, process allocated/reserved VRAM,
runtime, and exact artifact bytes.

The existing `public-evidence-gates-v1` thresholds are immutable. A custom
candidate can be described as successful only if it:

- improves primary RMSE by at least 2% over persistence and the stronger of
  MLP/TCN, and is directionally better than diagnostic LightGBM or is clearly
  described as not yet matching it;
- has a positive paired 24-hour block-bootstrap improvement interval using
  2,000 resamples and seed 20260730;
- improves in the same direction across all three neural seeds and does not
  materially regress bias, tail MAE, CRPS, or fold stability;
- keeps nominal 80% coverage within the immutable 70--90% gate;
- passes provenance, leakage, reproducibility, resource, and artifact checks.

Every evaluated condition is recorded as `PASS` or `FAIL`; an unavailable
condition is `NOT_EVALUATED`. `OPEN` is not a completed gate state. A null or
partial result is preserved exactly and cannot trigger confirmation retuning.

## Execution, evidence, and interfaces

Runs execute through Prefect and log to MLflow: repository/config/data/split
hashes, role and fold, seed, exact features, resolved config, environment,
runtime, total/process GPU memory, RAM, storage bytes, calibration record,
checkpoints, predictions, metrics, plots, OOD and component diagnostics,
failures, and retries. The API and console expose the chronological ledger,
ablation matrix, multi-fold/seed evidence, actual gate states, component
diagnostics, and a plain-language conclusion.

Every substantial experiment has LaTeX source, a compiled PDF, and visual
inspection. The final synthesis includes Cycle 0, provenance, leakage controls,
all ablations including negative results, confirmation, claim-to-evidence
mapping, resource use, limitations, and next phase.

## Budgets, safety, and stop rules

- Up to 12 wall-clock hours and 8 local GPU-hours; target at least 10 productive
  hours without idling merely to meet duration.
- One training process, `STRATA_NUM_WORKERS=0`, and at most four
  PyTorch/OpenMP/MKL/OpenBLAS CPU threads.
- Zero cloud dollars, no rentals, no gated data, no credentials, no champion
  promotion, and no autonomous merge.
- Preserve at least 50 GiB free on C:, add no more than 25 GiB, and never delete
  unrelated data or globally prune Docker.
- Do not start when CPU exceeds 85% for three minutes, free RAM is below
  12 GiB, free GPU memory is below 4 GiB, or GPU temperature exceeds 75 C.
- Gracefully stop below 8 GiB free RAM, above 15 GiB total board VRAM, above
  14 GiB process peak VRAM, or above 84 C for 60 seconds.
- One OOM recovery may halve batch size and double accumulation. A second OOM,
  provenance/manifest drift, leakage, corrupt labels, unsafe resources,
  permission/terms requirements, or inability to preserve evidence is a genuine
  blocker.
- At 2026-07-31 06:06 America/Chicago, start no new run or research revision.
  A bounded in-flight run may finish; afterwards perform evidence finalization
  only.

The goal requested updates to draft PR #2, but the user explicitly merged that
PR before this program began. This branch will therefore open a new draft PR
and will not merge it.
