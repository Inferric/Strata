# Phased execution plan

## Phase 0 — reproducible workstation

Deliver the Python/Node/Docker toolchain, CUDA validation, PostgreSQL, MLflow,
Prefect, DVC, tests, LaTeX compiler, API, and research console.

Exit criteria:

- repository validation, unit tests, and frontend build pass;
- CUDA sees the RTX 5080 and reports peak allocatable memory;
- MLflow and Prefect persist through service restarts;
- the console shows service health and no fabricated runs.

## Phase 1 — first direct-label result

Acquire `regression.mlo_cn2.dropna.Cn2_15m` through `otbench`, snapshot returned
bytes, hash the snapshot, freeze chronological blocks, and run climatology,
persistence, a compact MLP, and Strata-OT Surface.

Start with memory probes and fast-dev runs. Then use two seeds for each stable
neural candidate. The total local budget is four GPU-hours.

Exit criteria:

- all runs, checkpoints, metrics, peak VRAM, and resolved configs are in MLflow;
- split/preprocessing/provenance gates pass;
- Strata-OT is compared on identical data against all baselines;
- a compiled and inspected LaTeX PDF exists;
- measured runs are visible in the console.

## Phase 2 — reliable surface family

Extend direct-label tasks, add physical baselines and input ablations, evaluate
blocked seasons/campaigns/sites/instruments, calibrate uncertainty, and test
selective prediction under distribution shift.

The first Phase 2 validation is the leakage-safe one-step forecast described in
`docs/START_HERE.md`: MLO train/validation for development, followed by a frozen
one-shot USNA confirmation. It uses honest persistence, recent-mean, LightGBM,
compact MLP, and Strata-OT comparisons on identical six-row histories.

The next Phase 2 development experiment is the frozen MLO weather-horizon
ablation in `docs/NEXT_EXPERIMENT.md`. It holds back separate chronological
selection, calibration, and assessment roles within the official MLO
validation block; compares history-only and allowlisted operational-weather
inputs at 5--60 minutes; and tests the custom 2--5M-parameter Strata-OT Horizon
v1 without opening the official MLO test or retuning after assessment release.

That experiment is now Cycle 0 of the completed rolling-origin Fusion v2 program frozen
in `docs/FUSION_V2_PROGRAM.md`. Fusion v2 uses three seasonal USNA-large
development folds, multi-timescale context, deep weather conditioning,
training-only atmospheric pretraining, structured ablations, and a
preregistered one-shot 2022 confirmation. The MLO test stays sealed and no
model can be promoted by this autonomous program. Fusion v2.5 was the strongest
custom candidate but missed the frozen MLP-margin and tail-MAE conditions, so
confirmation remained sealed and the program closed without promotion.

The bounded public-profile adapter/QC step in
`docs/ESO_PARANAL_MASS_DATASET.md` is the next Phase 2-to-3 bridge. It may
materialize and verify one month of anonymous ESO Paranal MASS profiles, but it
does not authorize Column training. A multi-season purged profile protocol
must be frozen separately before Phase 3 model work.

Exit criteria: no promotion from a single seed/site/metric; tail and OOD
evidence are reported; inference latency and VRAM are measured.

## Phase 3 — vertical profile model

Download a bounded OTProf subset, reproduce the published baseline, and train
Strata-OT Column at 8–15M parameters. Teacher data are used for representation
pretraining only. Fine-tune/evaluate on independent profile observations once
terms and access are resolved.

Exit criteria: vertical gradient, layer peak, integrated optical, calibration,
and withheld-site/instrument metrics all exist; direct and teacher results are
never combined into one headline score.

## Phase 4 — heterogeneous observation operators

Add typed point/path/layer/integrated observations and the required geometry,
wavelength, response, and uncertainty metadata. Introduce latent
temperature–humidity covariance and a differentiable refractivity layer.

Exit criteria: every loss term maps to a documented observation operator and
all modality ablations are evaluated on sealed splits.

## Phase 5 — budgeted cloud scale-up

Move only validated, memory-profiled experiments to an approved SkyPilot job.
Start with one 48–80 GB GPU, use spot recovery when checkpoint semantics are
tested, and cap jobs by dollars, time, and count. Cloud credentials and spending
require explicit approval.

Exit criteria: automatic teardown, full cost logging, portable checkpoints, and
local reproduction of a reduced configuration.

## Phase 6 — public atmospheric-core release

Package model cards, data cards, BibTeX, code, frozen evaluation definitions,
weights whose data terms permit release, and LaTeX reports. Keep controlled
adapters and mission-specific integration out of the public project.
