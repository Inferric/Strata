# Reproducing the first real OTBench result

This procedure starts from a clean checkout and produces the real Mauna Loa
OTBench run, MLflow artifacts, Prefect flow record, console state, gate result,
best checkpoint selection, and compiled LaTeX report. It uses no cloud
resources, paid services, gated downloads, or synthetic labels.

## Prerequisites

- Python 3.11-3.13, `uv`, Node.js 22+, npm, Docker Desktop with Compose v2
- An NVIDIA GPU visible to WSL2/Windows; the reference run used one RTX 5080
- PowerShell 7 on Windows, or Bash on Linux/WSL2

Copy the environment template. Keep cloud access disabled. If local port 3000 is
already in use, select another console port before starting Compose.

```powershell
Copy-Item .env.example .env
# Optional:
# (Get-Content .env) -replace 'STRATA_WEB_PORT=3000', 'STRATA_WEB_PORT=3002' |
#   Set-Content .env
$env:STRATA_ALLOW_CLOUD = "false"
$env:STRATA_NUM_WORKERS = "0"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PREFECT_API_URL = "http://localhost:4200/api"
```

The Bash equivalents are:

```bash
cp .env.example .env
export STRATA_ALLOW_CLOUD=false
export STRATA_NUM_WORKERS=4
export MLFLOW_TRACKING_URI=http://localhost:5000
export PREFECT_API_URL=http://localhost:4200/api
```

## Install and validate the toolchain

```powershell
./scripts/bootstrap.ps1
uv sync --extra dev --extra data
nvidia-smi
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

On Bash, use `./scripts/bootstrap.sh`. The bootstrap performs the repository
validation, backend tests, frontend lint, and frontend production build.

## Start and check all services

```powershell
docker compose up -d postgres mlflow prefect api web
docker compose ps
uv run python scripts/wait_for_services.py
```

The default URLs are MLflow at <http://localhost:5000>, Prefect at
<http://localhost:4200>, the API at <http://localhost:8000/docs>, and the
console at `http://localhost:$STRATA_WEB_PORT` (3000 by default).

## Acquire and verify OTBench

The supported acquisition interface pins OTBench commit
`53cab9d53648b4870b43e35d48f19143786fa12e`, calls its public `TaskApi`, and
canonicalizes only the upstream-declared float32 log10 target transform.

```powershell
uv run strata-acquire --config configs/data/otbench_mlo.yaml
uv run strata-acquire --config configs/data/otbench_mlo.yaml --verify-only
```

If the immutable canonical snapshot already exists, omit the first command and
run verification only. The accepted dataset manifest SHA-256 for the reference
run is
`cd1bba8a0ca6ddc2a25f5d40a8f7b6d0187b48ad2c119d03d917c869980cf410`.
The materialized frozen split SHA-256 is
`6c6d817f5d26dcc8688e93f8e79bd183125491c5f5c7faf8a379182977e98940`.

## Run CUDA, loader, fast-development, and memory probes

These probes use real data and log their measured memory and runtime to MLflow.
They do not replace the full run.

```powershell
uv run strata-train --config configs/experiments/first_real_mlo.yaml --model mlp --seed 17 --fast-dev-run
uv run strata-train --config configs/experiments/first_real_mlo.yaml --model strata_ot_surface --seed 17 --fast-dev-run
```

The trainer enforces deterministic algorithms, BF16, math-only scaled
dot-product attention, the strict 15.5 GB gate, and the four-hour local GPU cap.
If an OOM occurs, the supported recovery is a smaller batch with proportionally
larger gradient accumulation; do not change the architecture or sealed evidence.

## Run the complete experiment

```powershell
uv run strata-first-flow --config configs/experiments/first_real_mlo.yaml
```

This single Prefect flow logs climatology, persistence, MLP seeds 17 and 41,
and Strata-OT Surface seeds 17 and 41. It evaluates the frozen gate and compiles
`reports/generated/first-real-run/experiment-report.pdf`.

The wrapper performs service startup, acquisition/verification, both probes,
and the full flow:

```powershell
./scripts/first-real-run.ps1
```

Use `./scripts/first-real-run.sh` on Bash.

## Verify the evidence and report

```powershell
uv run strata-acquire --config configs/data/otbench_mlo.yaml --verify-only
uv run strata-report --summary artifacts/latest/summary.json --output reports/generated/first-real-run
uv run strata-evaluate --summary artifacts/latest/summary.json
```

`strata-evaluate` exits with status 2 when a sealed gate fails. For the
reference result this is expected and records `interval_overcoverage`; it is not
a training crash. Inspect `artifacts/latest/gate-result.json` and do not convert
that exit into a pass.

The selected neural checkpoint is recorded in
`artifacts/latest/best-checkpoint.json`. All final runs include predictions,
diagnostic plots, environment/configuration evidence, runtime, peak VRAM, and
the exact source snapshot in MLflow.

## Closing validation

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
uv run python -m strata_ot.tools.validate_repo
npm --prefix web run lint
npm --prefix web run build
uv run strata-acquire --config configs/data/otbench_mlo.yaml --verify-only
uv run strata-report --summary artifacts/latest/summary.json --output reports/generated/first-real-run
uv run python scripts/wait_for_services.py
docker compose ps
```

The reference result is deliberately negative: persistence RMSE is 0.2751, the
best neural RMSE is 0.4008, and the formal calibration gate failed. The
next experiment must use validation-only scale calibration and a newly frozen
confirmatory test; the sealed test already reported here must not be reused for
tuning.
