#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.11-3.13 is required." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user "uv>=0.8,<1"
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker Engine with Compose v2 is required for the local research services." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js 22+ and npm are required for the research console." >&2
  exit 1
fi

mkdir -p artifacts/latest data/manifests reports/generated checkpoints
uv sync --extra dev --extra data
npm --prefix web ci

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA wheel cannot see the GPU'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
else
  echo "nvidia-smi was not found. Installation can be validated, but the real run must wait for the RTX 5080 CUDA environment." >&2
fi

uv run tectonic --version

uv run python -m strata_ot.tools.validate_repo
uv run pytest
npm --prefix web run lint
npm --prefix web run build
echo "Bootstrap complete. Start services with: docker compose up -d postgres mlflow prefect api web"
