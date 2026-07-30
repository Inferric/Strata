$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $PSScriptRoot
Set-Location $RepoDir

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11-3.13 is required."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    python -m pip install --user "uv>=0.8,<1"
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop with Compose v2 is required."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js 22+ and npm are required."
}

New-Item -ItemType Directory -Force artifacts/latest, data/manifests, reports/generated, checkpoints | Out-Null
uv sync --extra dev --extra data
npm --prefix web ci

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA wheel cannot see the GPU'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
} else {
    Write-Warning "nvidia-smi was not found; do not claim a real GPU result from this environment."
}

uv run tectonic --version

uv run python -m strata_ot.tools.validate_repo
uv run pytest
npm --prefix web run lint
npm --prefix web run build
Write-Host "Bootstrap complete. Start services with: docker compose up -d postgres mlflow prefect api web"
