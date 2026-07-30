---
name: cn2-research-pipeline
description: Run reproducible Cn² optical-turbulence research from approved dataset acquisition through physics-aware neural-model training, leakage-resistant evaluation, guarded autonomous experimentation, model promotion, and compiled LaTeX reporting. Use for Strata-OT, optical-turbulence datasets, Cn² benchmarks, High-Energy Laser atmospheric-effects research, first training runs, model sweeps, or semi-autonomous Codex research loops.
---

# Cn² Research Pipeline

Treat every run as a scientific experiment with immutable evidence, not as a leaderboard chase.

## Start

1. Locate the repository root and read `AGENTS.md`.
2. Read the repository's research brief, dataset registry, evaluation gates, and current experiment ledger.
3. Load only the reference needed for the current job:
   - Dataset work: `references/data-and-provenance.md`
   - Model or evaluation work: `references/model-and-evaluation.md`
   - Autonomous loops or reports: `references/autonomy-and-reporting.md`
4. Record assumptions in the run manifest before changing code or launching compute.

## Acquire data

1. Select a source from the checked-in registry.
2. Verify its access terms, expected files, checksums when published, citation, and redistribution status.
3. Use an automated adapter only for openly downloadable data. Stop for credentials, click-through terms, publication conditions, or ambiguous licenses.
4. Preserve raw bytes under an immutable content-addressed path. Never edit raw data in place.
5. Normalize into the repository's geometry-aware schema and emit a data card, QC summary, and frozen split identifier.
6. Keep direct observations, synthetic/teacher labels, derived turbulence proxies, and covariates distinguishable in every table and tensor.

## Design and run an experiment

1. State one falsifiable hypothesis and its comparison.
2. Choose the smallest model capable of testing it.
3. On a 16 GB development GPU, default to a 2–15M parameter model, BF16 mixed precision, gradient accumulation, activation checkpointing when useful, and a bounded data subset. Measure memory before scaling.
4. Compare against persistence or climatology, a physical parameterization, a conventional neural baseline, and the current champion when applicable.
5. Launch through the repository workflow so the run receives a unique ID, resolved configuration, code revision, data/split IDs, environment fingerprint, seed, checkpoints, metrics, plots, and cost record.
6. Log experiments and models to MLflow; use Prefect for orchestration and retries; use Optuna only inside an explicit trial and cost budget.

## Evaluate without leakage

1. Never use random row splits for primary evidence.
2. Apply blocked-time, leave-site, leave-campaign, leave-instrument, regime, and extreme-event tests as the available data supports.
3. Fit scalers, imputers, and calibration layers on training folds only.
4. Report log-space error, calibration, integrated optical metrics, layer/extreme skill, latency, memory, and selective/OOD risk. Do not report a single R² as the conclusion.
5. Fail promotion when coverage, provenance, split integrity, calibration, or reproducibility gates fail—even if an aggregate score improves.
6. Never let the autonomous agent edit sealed test data, split definitions, or promotion thresholds during an experiment cycle.

## Run the guarded research loop

1. Ask Codex for a structured experiment proposal, not unrestricted shell activity.
2. Validate the proposal with `scripts/check_run_contract.py`.
3. Limit each cycle to approved configuration, model, analysis, and report paths.
4. Execute one bounded experiment, evaluate it, compile its LaTeX report, then decide whether evidence supports another cycle.
5. Stop on exhausted budget, repeated non-improvement, failed data gates, numerical instability, unsafe hardware state, unavailable credentials, uncertain data rights, or a requested expansion of scope.
6. Require human approval before cloud provisioning, spending, dataset terms acceptance, champion promotion, public release, or any controlled-data interaction.

## Report

1. Generate scientific outputs from checked-in LaTeX source and compile them to PDF.
2. Treat JSON, CSV, Parquet, figures, and logs as supporting artifacts; do not substitute a Markdown-only final report.
3. Include provenance, exact configurations, data coverage, negative results, limitations, and a claim-to-evidence table.
4. Mark proposed architectures and inferred conclusions as proposals or inferences.
5. Keep public atmospheric science separate from controlled mission adapters, effects models, and integration details.

## Completion criteria

Finish only when the requested artifact is reproducible from committed configuration, the run manifest is complete, gates are evaluated, the LaTeX report compiles, and the dashboard can locate the run and its artifacts.
