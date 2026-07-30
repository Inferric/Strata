# Short-forecast validation: exact reproduction

Run these commands from the repository root in PowerShell. They use public data
only, zero cloud dollars, the frozen evaluation gate, and one 16 GB local GPU.

## 1. Install and start the local stack

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue
uv sync --extra dev --extra data
npm --prefix web ci
$env:STRATA_NUM_WORKERS = "0"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
docker compose up -d --build postgres mlflow prefect api web
uv run python scripts/wait_for_services.py
```

## 2. Validate code, proposals, CUDA, and the MLO development data

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
uv run python -m strata_ot.tools.validate_repo
npm --prefix web run lint
npm --prefix web run build
Get-ChildItem configs/proposals/*.json | ForEach-Object {
  uv run python scripts/check_run_contract.py $_.FullName
}
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30)"
uv run strata-acquire --config configs/data/otbench_mlo.yaml --verify-only
```

## 3. Run bounded MLO probes, then the full development experiment

These commands never load the MLO test partition.

```powershell
uv run strata-forecast --config configs/experiments/mlo_short_forecast_dev.yaml --model mlp --seed 17 --fast-dev-run
uv run strata-forecast --config configs/experiments/mlo_short_forecast_dev.yaml --model strata_ot_surface --seed 17 --fast-dev-run
uv run strata-forecast-flow --config configs/experiments/mlo_short_forecast_dev.yaml
```

Review the MLO summary and report before freezing the confirmation:

```powershell
Get-Content artifacts/latest/summary.json
```

## 4. Acquire and verify public USNA data without releasing test metrics

```powershell
uv run strata-acquire --config configs/data/otbench_usna_sm_forecast.yaml
uv run strata-acquire --config configs/data/otbench_usna_sm_forecast.yaml --verify-only
uv run strata-forecast --config configs/experiments/usna_short_forecast_confirm.yaml --model strata_ot_surface --seed 17 --fast-dev-run
```

Commit the code, frozen configuration, proposals, and USNA manifest before the
next command. Do not change model or evaluation settings after inspecting USNA
test output.

```powershell
git add src configs schemas tests reports/templates research docs README.md pyproject.toml uv.lock web
git commit -m "Add leakage-safe short-horizon Cn2 validation"
```

## 5. Release the official USNA test once

```powershell
uv run strata-forecast-flow --config configs/experiments/usna_short_forecast_confirm.yaml --release-test
```

The command trains the frozen MLP and Strata configurations for seeds 17 and 41,
fits uncertainty scale only on the validation calibration block, evaluates the
official USNA test, logs all artifacts to MLflow, evaluates the immutable gates,
and compiles the LaTeX report.

## 6. Final verification

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
uv run python -m strata_ot.tools.validate_repo
uv run strata-acquire --config configs/data/otbench_mlo.yaml --verify-only
uv run strata-acquire --config configs/data/otbench_usna_sm_forecast.yaml --verify-only
npm --prefix web run lint
npm --prefix web run build
uv run strata-evaluate --summary artifacts/latest/summary.json --gates configs/evaluation/gates/public_v1.yaml
docker compose ps
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/api/overview
Invoke-WebRequest http://localhost:5000/health
Invoke-WebRequest http://localhost:4200/api/health
Invoke-WebRequest http://localhost:3002
```

The final PDF is
`reports/generated/short-forecast-validation/forecast-report.pdf`. Raw data,
sealed row identities, checkpoints, run artifacts, and generated reports are
locally preserved but ignored by Git unless explicitly released.

Do not rerun the USNA test to select a different model. The next change should
use a new development split or a newly registered external confirmation task.
