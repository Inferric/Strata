from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from mlflow import MlflowClient

from strata_ot.config import find_repo_root


def _allowed_web_origins() -> list[str]:
    configured = os.getenv(
        "STRATA_WEB_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(title="Strata-OT Research API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_web_origins(),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _root_or_app() -> Path:
    try:
        return find_repo_root()
    except FileNotFoundError:
        return Path("/app")


def _serialize_run(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.info.run_id,
        "name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
        "status": run.info.status,
        "started_at": run.info.start_time,
        "model": run.data.tags.get("model_family", "baseline"),
        "kind": run.data.tags.get("run_kind", "unknown"),
        "dataset_id": run.data.tags.get("dataset_id"),
        "split_id": run.data.tags.get("split_id"),
        "site": run.data.tags.get("site"),
        "task_kind": run.data.tags.get("task_kind"),
        "feature_set": run.data.tags.get("feature_set"),
        "horizon_rows": run.data.tags.get("horizon_rows"),
        "horizon_minutes": run.data.tags.get("horizon_minutes"),
        "forecast_horizon_minutes": run.data.tags.get("forecast_horizon_minutes"),
        "evaluation_partition": run.data.tags.get("evaluation_partition"),
        "metrics": run.data.metrics,
        "parameters": run.data.params,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
async def runs() -> dict[str, Any]:
    try:
        client = MlflowClient(tracking_uri=_tracking_uri())
        experiments = client.search_experiments(max_results=100)
        experiment_ids = [experiment.experiment_id for experiment in experiments]
        records = (
            client.search_runs(
                experiment_ids,
                max_results=100,
                order_by=["attributes.start_time DESC"],
            )
            if experiment_ids
            else []
        )
        return {"source": "mlflow", "items": [_serialize_run(run) for run in records]}
    except Exception as error:
        return {"source": "unavailable", "items": [], "message": str(error)}


@app.get("/api/datasets")
async def datasets() -> dict[str, Any]:
    root = _root_or_app()
    manifests = root / "data" / "manifests"
    items: list[dict[str, Any]] = []
    if manifests.exists():
        for path in sorted(manifests.glob("*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                qc_path = (
                    root / "artifacts" / "data-qc" / f"{item.get('dataset_id', path.stem)}.json"
                )
                if qc_path.is_file():
                    qc = json.loads(qc_path.read_text(encoding="utf-8"))
                    checks = qc.get("checks", {})
                    item["verification"] = {
                        "manifest_verified": qc.get("manifest_verified", False),
                        "checks_passed": sum(bool(value) for value in checks.values()),
                        "checks_total": len(checks),
                        "split_id": qc.get("split", {}).get("id"),
                        "split_sha256": qc.get("split", {}).get("sealed_sha256"),
                    }
                items.append(item)
            except json.JSONDecodeError:
                items.append({"dataset_id": path.stem, "status": "invalid_manifest"})
    return {"items": items}


@app.get("/api/overview")
async def overview() -> dict[str, Any]:
    run_payload = await runs()
    run_items = run_payload["items"]
    completed = [item for item in run_items if item["status"] == "FINISHED"]
    best = min(
        (
            item
            for item in completed
            if "test/rmse_log10_cn2" in item["metrics"] or "rmse_log10_cn2" in item["metrics"]
        ),
        key=lambda item: item["metrics"].get(
            "rmse_log10_cn2", item["metrics"].get("test/rmse_log10_cn2", float("inf"))
        ),
        default=None,
    )
    summary_path = _root_or_app() / "artifacts" / "latest" / "summary.json"
    latest = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return {
        "run_count": len(run_items),
        "completed_count": len(completed),
        "best_run": best,
        "latest_summary": latest,
        "tracking_source": run_payload["source"],
    }


@app.get("/api/horizon")
async def horizon() -> dict[str, Any]:
    summary_path = _root_or_app() / "artifacts" / "latest" / "summary.json"
    if not summary_path.is_file():
        raise HTTPException(status_code=404, detail="No completed horizon experiment")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("task_kind") != "multi_horizon_forecast":
        raise HTTPException(status_code=404, detail="Latest experiment is not multi-horizon")
    return {
        "experiment_id": summary["experiment_id"],
        "evaluation_partition": summary["evaluation_partition"],
        "assessment_released": summary["assessment_released"],
        "plain_language_conclusion": summary["plain_language_conclusion"],
        "horizon_matrix": summary["horizon_matrix"],
        "component_diagnostics": [
            {
                "run_id": run["run_id"],
                "model": run["model"],
                "feature_set": run["feature_set"],
                "seed": run.get("seed"),
                "by_horizon": run.get("component_summary", {}),
            }
            for run in summary["runs"]
            if run["model"] == "strata_ot_horizon"
        ],
        "assessment_claim": summary["assessment_claim"],
        "checks": summary["checks"],
        "gate_result": summary.get("gate_result"),
    }


@app.get("/api/fusion")
async def fusion() -> dict[str, Any]:
    root = _root_or_app()
    program_root = root / "artifacts" / "experiments" / "strata-fusion-v2-program"
    candidates = (
        program_root / "program-summary.json",
        program_root / "robustness-summary.json",
        program_root / "screen-summary.json",
    )
    summary_path = next((path for path in candidates if path.is_file()), None)
    if summary_path is None:
        raise HTTPException(
            status_code=404,
            detail="No completed Fusion v2 program evidence",
        )
    summary = cast(
        dict[str, Any],
        json.loads(summary_path.read_text(encoding="utf-8")),
    )
    if summary.get("task_kind") not in {
        "fusion_v2_screen",
        "fusion_v2_robustness",
        "fusion_v2_program",
    }:
        raise HTTPException(
            status_code=404,
            detail="Latest Fusion artifact has an unsupported task kind",
        )
    return summary


@app.get("/api/reports/latest", response_class=FileResponse)
async def latest_report() -> FileResponse:
    root = _root_or_app()
    summary_candidates = (
        root / "artifacts" / "experiments" / "strata-fusion-v2-program" / "program-summary.json",
        root / "artifacts" / "latest" / "summary.json",
    )
    summary_path = next(
        (path for path in summary_candidates if path.is_file()),
        None,
    )
    if summary_path is None:
        raise HTTPException(status_code=404, detail="No completed report")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_path = summary.get("report_pdf")
    if not isinstance(report_path, str):
        raise HTTPException(status_code=404, detail="No completed report")
    candidate = (root / report_path.replace("\\", "/")).resolve()
    reports_root = (root / "reports" / "generated").resolve()
    if not candidate.is_relative_to(reports_root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Report artifact is unavailable")
    return FileResponse(
        candidate,
        media_type="application/pdf",
        filename=candidate.name,
    )


@app.get("/api/reports/cycle/{cycle_id}", response_class=FileResponse)
async def cycle_report(cycle_id: str) -> FileResponse:
    root = _root_or_app()
    summary_path = (
        root / "artifacts" / "experiments" / "strata-fusion-v2-program" / "program-summary.json"
    )
    if not summary_path.is_file():
        raise HTTPException(status_code=404, detail="No program synthesis")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cycle = next(
        (
            item
            for item in summary.get("cycles", [])
            if isinstance(item, dict) and item.get("cycle_id") == cycle_id
        ),
        None,
    )
    report_path = cycle.get("report_pdf") if isinstance(cycle, dict) else None
    if not isinstance(report_path, str):
        raise HTTPException(status_code=404, detail="Cycle report unavailable")
    candidate = (root / report_path.replace("\\", "/")).resolve()
    reports_root = (root / "reports" / "generated").resolve()
    if not candidate.is_relative_to(reports_root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Cycle report unavailable")
    return FileResponse(
        candidate,
        media_type="application/pdf",
        filename=candidate.name,
    )


@app.get("/api/system")
async def system() -> dict[str, Any]:
    prefect_url = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
    prefect_status = "unavailable"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{prefect_url.rstrip('/')}/health")
            prefect_status = "healthy" if response.is_success else "degraded"
    except httpx.HTTPError:
        pass
    return {
        "api": "healthy",
        "prefect": prefect_status,
        "mlflow": (await runs())["source"],
        "hardware_target": "RTX 5080 / 16 GB",
        "cloud_allowed": os.getenv("STRATA_ALLOW_CLOUD", "false").lower() == "true",
        "cloud_budget_usd": float(os.getenv("STRATA_MAX_CLOUD_COST_USD", "0")),
    }
