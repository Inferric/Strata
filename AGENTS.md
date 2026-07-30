# Agent operating contract

## Mission

Build reproducible public-source Cn² research infrastructure and neural models. Prefer evidence over model size and a bounded experiment over an open-ended sweep.

## Required skills

Use the project-local skills that match the work. For an end-to-end run, explicitly invoke `$cn2-research-pipeline`. Use the MLflow, PyTorch Lightning, literature, LaTeX, cloud, and frontend skills only for their focused jobs.

## Reports

- Generate every completed scientific, experiment, benchmark, dataset-QC, and milestone report from LaTeX source.
- Compile to PDF and inspect the result before marking work complete.
- Markdown may be used for working notes and documentation, never as the sole final scientific report.
- Cite primary sources and preserve BibTeX.

## Data

- Never edit raw data in place.
- Never automate login-gated or click-through dataset acquisition.
- Record source, citation, terms, checksums, geometry, instrument, wavelength, units, coverage, QC, and label provenance.
- Keep direct observations, synthetic teachers, derived proxies, and covariates explicitly separated.
- Fit preprocessing on training folds only.

## Evaluation

- Random row splits are prohibited as primary evidence.
- Use task-defined or frozen blocked-time/grouped splits.
- Group colocated sensors and overlapping paths.
- Report log-space error, distribution calibration, tail/layer behavior, integrated optical metrics when supported, OOD/selective risk, latency, VRAM, and cost.
- Do not promote a model from one seed or one scalar metric.

## Autonomy

- Treat `configs/splits/frozen/`, `configs/evaluation/gates/`, `data/raw/`, and `data/sealed/` as immutable.
- Validate autonomous proposals against `schemas/experiment_proposal.schema.json`.
- Default local budget is zero cloud dollars and four GPU-hours per experiment.
- Stop for credentials, gated terms, public-release questions, cloud spending, champion promotion, or controlled-data interaction.
- Do not expose credentials to repository-controlled lifecycle hooks.

## Local hardware

Assume one RTX 5080 with 16 GB VRAM unless the active run manifest says otherwise. Start with BF16, a bounded sample, conservative worker counts, and measured peak memory. Scale model width, context, resolution, or data only after the smaller run is valid.

## Commands

```bash
uv sync --extra dev --extra data
uv run ruff check .
uv run mypy src
uv run pytest
uv run python -m strata_ot.tools.validate_repo
npm --prefix web run lint
npm --prefix web run build
```

## Completion

Work is complete only when relevant tests pass, the run/config/data identities are recorded, required gates are evaluated, artifacts are visible through the API/console, and the LaTeX PDF compiles.
