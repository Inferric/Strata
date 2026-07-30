#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ "${STRATA_ALLOW_CLOUD:-false}" == "true" ]]; then
  echo "The first run is local-only. Set STRATA_ALLOW_CLOUD=false." >&2
  exit 1
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export PREFECT_API_URL="${PREFECT_API_URL:-http://localhost:4200/api}"
export STRATA_NUM_WORKERS="${STRATA_NUM_WORKERS:-4}"

docker compose up -d postgres mlflow prefect api web
docker compose ps
uv run python scripts/wait_for_services.py

if [[ ! -f data/raw/otbench/canonical-v1/otbench-mlo-cn2-15m-v1/train_features.csv ]]; then
  uv run strata-acquire --config configs/data/otbench_mlo.yaml
fi
uv run strata-acquire --config configs/data/otbench_mlo.yaml --verify-only

uv run strata-train --config configs/experiments/first_real_mlo.yaml --model mlp --seed 17 --fast-dev-run
uv run strata-train --config configs/experiments/first_real_mlo.yaml --model strata_ot_surface --seed 17 --fast-dev-run
uv run strata-first-flow --config configs/experiments/first_real_mlo.yaml
uv run python -c "import json; s=json.load(open('artifacts/latest/summary.json')); print('gate:', 'PASS' if s['gate_result']['passed'] else 'OPEN ' + ','.join(s['gate_result']['failures']))"
echo "First real run complete. Open http://localhost:${STRATA_WEB_PORT:-3000} and reports/generated/first-real-run/experiment-report.pdf"
