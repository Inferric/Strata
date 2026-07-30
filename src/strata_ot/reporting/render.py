from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from strata_ot.config import find_repo_root

LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: object) -> str:
    text = str(value)
    return "".join(LATEX_REPLACEMENTS.get(character, character) for character in text)


def latex_number(value: object, digits: int = 4) -> str:
    if value is None:
        return "--"
    if not isinstance(value, (str, int, float)):
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def latex_percent(value: object, digits: int = 1) -> str:
    if value is None:
        return "--"
    if not isinstance(value, (str, int, float)):
        return "--"
    try:
        return f"{100.0 * float(value):.{digits}f}\\%"
    except (TypeError, ValueError):
        return "--"


def short_identity(value: object, length: int = 16) -> str:
    return str(value)[:length]


def _environment(template_dir: Path) -> Environment:
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        autoescape=False,
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
    )
    environment.filters["latex"] = latex_escape
    environment.filters["num"] = latex_number
    environment.filters["pct"] = latex_percent
    environment.filters["shortid"] = short_identity
    return environment


def _compile(tex_path: Path, output_dir: Path) -> Path:
    environment_tectonic = Path(sys.executable).with_name("tectonic")
    tectonic = shutil.which("tectonic")
    if tectonic is None and environment_tectonic.is_file():
        tectonic = str(environment_tectonic)
    latexmk = shutil.which("latexmk")
    if tectonic:
        command = [tectonic, "-X", "compile", tex_path.name, "--outdir", str(output_dir)]
    elif latexmk:
        command = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={output_dir}",
            tex_path.name,
        ]
    else:
        raise RuntimeError("Install Tectonic or latexmk; PDF is a required run artifact")
    completed = subprocess.run(
        command,
        cwd=tex_path.parent,
        check=False,
        text=True,
        capture_output=True,
    )
    (output_dir / "compile.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "compile.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"LaTeX compilation failed; see {output_dir}/compile.stderr.txt")
    pdf = output_dir / f"{tex_path.stem}.pdf"
    if not pdf.exists() or pdf.stat().st_size < 1000:
        raise RuntimeError("LaTeX compiler returned success without a usable PDF")
    return pdf


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], payload)


def _failure_summary(payload: dict[str, Any]) -> str:
    failure = payload.get("failure", {})
    message = str(failure.get("message", ""))
    if "baseline_value" in message and "Changing param values" in message:
        return (
            "MLflow rejected a duplicate Lightning hyperparameter with inconsistent "
            "None serialization. The duplicate manual log was removed before the final flow."
        )
    if "charmap" in message and "codec can't encode" in message:
        return (
            "MLflow finalization encountered the Windows CP1252 console. Standard output "
            "was explicitly configured as UTF-8 and the probe retry completed."
        )
    first_line = message.splitlines()[0] if message else "No message recorded"
    return first_line[:300]


def _prepare_report_context(
    summary: dict[str, Any],
    root: Path,
    output_dir: Path,
) -> None:
    data_qc_path = root / str(summary["data_qc"])
    summary["data_qc_details"] = _load_json(data_qc_path)
    summary["resolved_configuration"]["model"].pop(
        "effective_input_dim_from_data",
        None,
    )
    summary["resolved_configuration"]["model"]["effective_input_dim"] = (
        summary["data_qc_details"]["upstream_task"]["features"]
    )

    completed = [
        run
        for run in summary.get("runs", [])
        if isinstance(run.get("metrics", {}).get("rmse_log10_cn2"), (int, float))
    ]
    summary["best_overall_run"] = min(
        completed,
        key=lambda run: float(run["metrics"]["rmse_log10_cn2"]),
    )
    neural = [run for run in completed if run.get("kind") == "neural"]
    summary["best_neural_run"] = min(
        neural,
        key=lambda run: float(run["metrics"]["rmse_log10_cn2"]),
    )
    summary["peak_vram_gb"] = max(
        float(run["metrics"].get("peak_vram_gb", 0.0))
        for run in neural
    )
    summary["strata_interval_coverages"] = [
        float(run["metrics"]["interval_80_coverage"])
        for run in neural
        if run.get("model") == "strata_ot_surface"
        and run["metrics"].get("interval_80_coverage") is not None
    ]

    best_checkpoint_run = next(
        run
        for run in neural
        if run["run_id"] == summary["best_neural_run"]["run_id"]
    )
    checkpoint_path = Path(str(best_checkpoint_run["checkpoint"])).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is unavailable: {checkpoint_path}")
    checkpoint_record = {
        "run_id": best_checkpoint_run["run_id"],
        "model": best_checkpoint_run["model"],
        "seed": best_checkpoint_run["seed"],
        "rmse_log10_cn2": best_checkpoint_run["metrics"]["rmse_log10_cn2"],
        "path": checkpoint_path.relative_to(root).as_posix(),
        "display_path": "/".join(checkpoint_path.parts[-4:]),
        "sha256": _sha256_file(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
    }
    best_checkpoint_path = root / "artifacts" / "latest" / "best-checkpoint.json"
    best_checkpoint_path.write_text(
        json.dumps(checkpoint_record, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["best_checkpoint"] = checkpoint_record

    source_digest = summary["repository_identity"]["source_digest_sha256"]
    source_archive = root / "artifacts" / "source-snapshots" / f"{source_digest}.zip"
    if not source_archive.is_file():
        raise FileNotFoundError(f"Recorded source snapshot is unavailable: {source_archive}")
    summary["source_snapshot"] = {
        "path": source_archive.relative_to(root).as_posix(),
        "sha256": _sha256_file(source_archive),
        "bytes": source_archive.stat().st_size,
    }

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_models = ("persistence", "mlp", "strata_ot_surface")
    report_figures: list[dict[str, str]] = []
    for model in figure_models:
        candidates = [run for run in completed if run.get("model") == model]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda run: float(run["metrics"]["rmse_log10_cn2"]),
        )
        source = root / str(selected["artifacts"]["figure"])
        destination = figure_dir / f"{model}-diagnostics.png"
        shutil.copy2(source, destination)
        report_figures.append(
            {
                "model": model,
                "run_id": selected["run_id"],
                "name": selected["name"],
                "path": destination.relative_to(output_dir).as_posix(),
            }
        )
    summary["report_figures"] = report_figures

    failure_records: list[dict[str, str]] = []
    acquisition_failures = root / "artifacts" / "acquisition-failures"
    for path in sorted(acquisition_failures.glob("*.json")):
        payload = _load_json(path)
        failure_records.append(
            {
                "phase": "acquisition",
                "identity": payload.get("dataset_id", path.stem),
                "outcome": str(payload.get("status", "failed")).replace("_", " "),
                "explanation": str(payload.get("reason", "Acquisition rejected")),
                "evidence": path.relative_to(root).as_posix(),
            }
        )
    for path in sorted((root / "artifacts" / "runs").glob("*/failure-manifest.json")):
        payload = _load_json(path)
        failure_records.append(
            {
                "phase": str(payload.get("kind", "run")),
                "identity": str(payload.get("run_id", path.parent.name)),
                "outcome": str(payload.get("failure", {}).get("type", "failed")).replace(
                    "_", " "
                ),
                "explanation": _failure_summary(payload),
                "evidence": path.relative_to(root).as_posix(),
            }
        )
    summary["failure_records"] = failure_records
    summary["recommended_next_experiment"] = (
        "Before evaluating on a newly frozen confirmatory test, change one factor: "
        "replace the hard-clamped Gaussian scale with a smooth positive scale and fit "
        "a validation-only scale calibrator. Re-run the MLP and Strata-OT Surface for "
        "at least two seeds, retain the current blocked partitioning and purge, and "
        "treat irregular temporal gaps explicitly. The current sealed test must not "
        "be reused for calibration or hyperparameter selection."
    )


def _prepare_forecast_report_context(
    summary: dict[str, Any],
    root: Path,
    output_dir: Path,
) -> None:
    summary["data_qc_details"] = _load_json(root / str(summary["data_qc"]))
    development_path = (
        root
        / "artifacts"
        / "experiments"
        / "mlo-short-forecast-dev-v1"
        / "summary.json"
    )
    if (
        summary.get("experiment_id") != "mlo-short-forecast-dev-v1"
        and development_path.is_file()
    ):
        summary["development_summary"] = _load_json(development_path)
    completed = [
        run
        for run in summary.get("runs", [])
        if isinstance(run.get("metrics", {}).get("rmse_log10_cn2"), (int, float))
    ]
    if not completed:
        raise ValueError("A forecast report requires completed measured runs")
    summary["best_overall_run"] = min(
        completed,
        key=lambda run: float(run["metrics"]["rmse_log10_cn2"]),
    )
    neural = [run for run in completed if run.get("kind") == "neural"]
    if not neural:
        raise ValueError("A forecast report requires a completed neural run")
    summary["best_neural_run"] = min(
        neural,
        key=lambda run: float(run["metrics"]["rmse_log10_cn2"]),
    )
    checkpoint_path = Path(str(summary["best_neural_run"]["checkpoint"])).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is unavailable: {checkpoint_path}")
    checkpoint_record = {
        "run_id": summary["best_neural_run"]["run_id"],
        "model": summary["best_neural_run"]["model"],
        "seed": summary["best_neural_run"]["seed"],
        "rmse_log10_cn2": summary["best_neural_run"]["metrics"][
            "rmse_log10_cn2"
        ],
        "path": checkpoint_path.relative_to(root).as_posix(),
        "display_path": "/".join(checkpoint_path.parts[-4:]),
        "sha256": _sha256_file(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
    }
    best_checkpoint_path = root / "artifacts" / "latest" / "best-checkpoint.json"
    best_checkpoint_path.write_text(
        json.dumps(checkpoint_record, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["best_checkpoint"] = checkpoint_record
    source_digest = summary["repository_identity"]["source_digest_sha256"]
    source_archive = root / "artifacts" / "source-snapshots" / f"{source_digest}.zip"
    if not source_archive.is_file():
        raise FileNotFoundError(f"Recorded source snapshot is unavailable: {source_archive}")
    summary["source_snapshot"] = {
        "path": source_archive.relative_to(root).as_posix(),
        "sha256": _sha256_file(source_archive),
        "bytes": source_archive.stat().st_size,
    }
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_figures: list[dict[str, str]] = []
    for model in ("persistence", "lightgbm", "mlp", "strata_ot_surface"):
        candidates = [run for run in completed if run.get("model") == model]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda run: float(run["metrics"]["rmse_log10_cn2"]),
        )
        source = root / str(selected["artifacts"]["figure"])
        destination = figure_dir / f"{model}-diagnostics.png"
        shutil.copy2(source, destination)
        report_figures.append(
            {
                "model": model,
                "run_id": selected["run_id"],
                "name": selected["name"],
                "path": destination.relative_to(output_dir).as_posix(),
            }
        )
    summary["report_figures"] = report_figures
    failures: list[dict[str, str]] = []
    for path in sorted((root / "artifacts" / "runs").glob("*/failure-manifest.json")):
        payload = _load_json(path)
        resolved = payload.get("resolved_configuration", {})
        if resolved.get("task_kind") != "one_step_forecast":
            continue
        failures.append(
            {
                "phase": str(payload.get("kind", "run")),
                "identity": str(payload.get("run_id", path.parent.name)),
                "outcome": str(payload.get("failure", {}).get("type", "failed")),
                "explanation": _failure_summary(payload),
            }
        )
    summary["failure_records"] = failures
    summary["best_neural_improvement_percent"] = (
        100
        * float(summary["forecast_claim"].get("measured_relative_improvement", 0))
    )


def render_report(summary_path: Path, output_dir: Path) -> Path:
    root = find_repo_root(summary_path.parent)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    is_forecast = summary.get("task_kind") == "one_step_forecast"
    if is_forecast:
        _prepare_forecast_report_context(summary, root, output_dir)
    else:
        _prepare_report_context(summary, root, output_dir)
    template_dir = root / "reports" / "templates"
    template_name = (
        "forecast_report.tex.j2" if is_forecast else "experiment_report.tex.j2"
    )
    template = _environment(template_dir).get_template(template_name)
    summary.setdefault("checks", {})["latex_pdf_compiled"] = True
    tex = template.render(summary=summary)
    tex_path = output_dir / (
        "forecast-report.tex" if is_forecast else "experiment-report.tex"
    )
    tex_path.write_text(tex, encoding="utf-8")
    shutil.copy2(root / "research" / "references.bib", output_dir / "references.bib")
    try:
        pdf = _compile(tex_path, output_dir)
    except Exception:
        summary["checks"]["latex_pdf_compiled"] = False
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise
    summary["report_pdf"] = pdf.relative_to(root).as_posix()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Strata-OT LaTeX report")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = find_repo_root()
    summary = Path(args.summary)
    output = Path(args.output)
    if not summary.is_absolute():
        summary = root / summary
    if not output.is_absolute():
        output = root / output
    print(render_report(summary, output))


if __name__ == "__main__":
    main()
