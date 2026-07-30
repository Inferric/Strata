# Start here: what Strata-OT is doing

`Cn²` (written \(C_n^2\)) is a number describing how strongly the atmosphere
distorts light. Hot/cold air mixing, humidity, wind, and surface heating all
change it. This repository asks whether a small neural network can forecast that
number from public weather and Cn² measurements.

## The current experiment in one sentence

Given the last six measurements, predict the next Cn² measurement a few minutes
ahead, and beat the honest strategy of simply repeating the latest measurement.

For Mauna Loa, six rows cover about 30 minutes and the forecast is 5 minutes
ahead. For USNA, six rows cover about 36 minutes and the forecast is 6 minutes
ahead.

## Why there are several models

Think of them as increasingly flexible contestants that receive the same
information:

1. **Climatology** always says “the usual training value.”
2. **Persistence** says “the next value will equal the latest value.”
3. **Recent mean** averages the last six Cn² readings.
4. **LightGBM** is a strong conventional tree model.
5. **MLP** is a compact, ordinary neural network.
6. **Strata-OT Surface** is the research model. It uses attention to read the
   ordered history and a small regime router to represent different atmospheric
   conditions.

Persistence is important. The atmosphere is continuous, so a measurement from
five minutes ago is often already an excellent forecast. A neural model that
cannot beat it has not yet learned enough useful change.

## Why MLO comes before USNA

Mauna Loa (MLO) is the development site. We use its training and validation
blocks to fix bugs, choose the exact model, and calibrate uncertainty. Its old
test result is not reused.

USNA is the confirmation site. Its official test block stays sealed while the
configuration is developed. After the code and configuration are committed, we
release that test once. This protects us from unconsciously tuning to the answer.

## What “no leakage” means

Future information must not sneak into training. Strata-OT therefore:

- preserves OTBench's chronological train, validation, and test blocks;
- removes 24 rows around boundaries;
- rejects a six-row history if a time gap is too large;
- learns missing-value replacements and scaling from training only;
- gives persistence and both neural networks the same Cn² history;
- fits uncertainty calibration only on a validation calibration block.

## What the metrics mean

- **RMSE / MAE:** typical forecast error in log10 Cn²; lower is better.
- **Bias:** whether forecasts are systematically too high or low; near zero is
  better.
- **80% coverage:** an 80% uncertainty interval should contain about 80% of
  observations. Far above 80% means vague intervals; far below means
  overconfidence.
- **CRPS / NLL:** scores for the complete probability forecast; lower is better.
- **Tail MAE:** error on the strongest-turbulence tenth of observations.

A difference of roughly 0.3 log10 units is about a factor of two in the original
physical scale.

## What all the infrastructure is for

- **PostgreSQL** stores durable MLflow and Prefect records.
- **MLflow** is the experiment ledger: metrics, exact configuration,
  checkpoints, predictions, plots, environment, runtime, and failures.
- **Prefect** executes the ordered workflow and records retries.
- **Research API** exposes those real records to software.
- **Research console** turns them into a readable dashboard.
- **LaTeX report** is the final scientific record. It maps each claim to saved
  evidence and states limitations.

None of these services invent a metric. If a run has not happened, the console
shows no result.

## How to read the final verdict

First ask whether the best neural model beats persistence by at least 2%. Then
look at the 24-hour block-bootstrap interval: if it crosses zero, the apparent
improvement could plausibly be sampling noise. Finally check calibration,
seed-to-seed stability, leakage/provenance checks, VRAM, and runtime.

Passing those checks supports only this narrow claim: a model forecasted this
public site and horizon under this frozen evaluation. It does not establish
deployment readiness, causal physics, other seasons or sites, vertical-profile
accuracy, or long-range optical-link performance.

Exact reproduction commands are in
[`SHORT_FORECAST_RUN.md`](SHORT_FORECAST_RUN.md).
