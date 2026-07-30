from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from strata_ot.autonomy.contracts import load_and_validate
from strata_ot.config import find_repo_root, load_yaml
from strata_ot.orchestration.flows import candidate_flow


def _proposal_prompt(summary_path: Path) -> str:
    return f"""
Use $cn2-research-pipeline. Read AGENTS.md, the sealed evaluation gates, and
{summary_path}. Propose exactly one bounded follow-up experiment as JSON matching
schemas/experiment_proposal.schema.json.

The proposal must isolate a scientific hypothesis, use an existing public dataset
manifest and frozen split, change no more than three allowlisted model or optimizer
parameters, use at most two local runs and four local RTX 5080 GPU-hours, and set
cloud cost to zero. Stop conditions must include budget, NaN/divergence, OOM, and
data/split failure. Do not modify files, train, promote, or claim results.
""".strip()


def request_codex_proposal(root: Path, summary_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = output_dir / "proposal.json"
    events_path = output_dir / "codex-events.jsonl"
    command = [
        "codex",
        "exec",
        "--cd",
        str(root),
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--json",
        "--output-schema",
        str(root / "schemas" / "experiment_proposal.schema.json"),
        "--output-last-message",
        str(proposal_path),
        _proposal_prompt(summary_path.relative_to(root)),
    ]
    allowed_environment = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TERM",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "CODEX_HOME",
        "CODEX_API_KEY",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
    }
    environment = {
        key: value for key, value in os.environ.items() if key in allowed_environment
    }
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    events_path.write_text(completed.stdout, encoding="utf-8")
    (output_dir / "codex-stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Codex proposal failed; see {output_dir}/codex-stderr.txt")
    load_and_validate(proposal_path)
    return proposal_path


def materialize_experiment(
    root: Path,
    proposal: dict[str, Any],
    output_dir: Path,
    *,
    remaining_gpu_hours: float,
) -> Path:
    base_model_path = root / str(proposal["model"]["config_path"])
    model_config = load_yaml(base_model_path, root=root)
    trainer_config = load_yaml("configs/trainer/local_16gb.yaml", root=root)
    for change in proposal["changes"]:
        parameter = str(change["parameter"])
        if parameter in {"learning_rate", "weight_decay"}:
            trainer_config[parameter] = change["new"]
        else:
            model_config[parameter] = change["new"]
    trainer_config["max_gpu_hours"] = min(
        float(proposal["budget"]["max_local_gpu_hours"]),
        remaining_gpu_hours,
    )
    model_path = output_dir / "model.yaml"
    trainer_path = output_dir / "trainer.yaml"
    model_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")
    trainer_path.write_text(yaml.safe_dump(trainer_config, sort_keys=False), encoding="utf-8")
    experiment = {
        "id": proposal["proposal_id"],
        "hypothesis": proposal["hypothesis"],
        "seed": 17,
        "seeds": [17, 41][: int(proposal["budget"]["max_runs"])],
        "data": "configs/data/otbench_mlo.yaml",
        "model": str(model_path.relative_to(root)),
        "trainer": str(trainer_path.relative_to(root)),
        "evaluation": "configs/evaluation/gates/public_v1.yaml",
        "models": [proposal["model"]["family"]],
        "tracking": {
            "experiment_name": "strata-ot-autonomous-candidates",
            "register_candidate": False,
        },
        "report": {
            "template": "reports/templates/experiment_report.tex.j2",
            "output_dir": f"reports/generated/{proposal['proposal_id']}",
        },
    }
    experiment_path = output_dir / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    return experiment_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded Codex research iteration")
    parser.add_argument("--summary", default="artifacts/latest/summary.json")
    parser.add_argument("--execute", action="store_true", help="Train after proposal validation")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum proposal/train/report cycles in this invocation (1-10)",
    )
    parser.add_argument(
        "--max-total-gpu-hours",
        type=float,
        default=4.0,
        help="Hard local budget shared by all cycles in this invocation (maximum 4)",
    )
    args = parser.parse_args()
    if not 1 <= args.max_iterations <= 10:
        raise SystemExit("--max-iterations must be between 1 and 10")
    if not 0 < args.max_total_gpu_hours <= 4:
        raise SystemExit("--max-total-gpu-hours must be greater than zero and at most four")
    os.environ.setdefault("PREFECT_API_URL", "http://localhost:4200/api")
    root = find_repo_root()
    summary = root / args.summary
    if not summary.exists():
        raise SystemExit("Complete the first real run before starting the iterative loop")
    proposals_root = root / "artifacts" / "proposals"
    proposals_root.mkdir(parents=True, exist_ok=True)
    iteration = max(
        (
            int(path.name.removeprefix("iteration-"))
            for path in proposals_root.glob("iteration-*")
            if path.is_dir() and path.name.removeprefix("iteration-").isdigit()
        ),
        default=0,
    )
    gpu_hours_used = 0.0
    for _cycle in range(args.max_iterations):
        remaining_gpu_hours = args.max_total_gpu_hours - gpu_hours_used
        if remaining_gpu_hours <= 0:
            print("Stopped: the shared local GPU-hour budget is exhausted")
            break
        iteration += 1
        output_dir = proposals_root / f"iteration-{iteration:03d}"
        proposal_path = request_codex_proposal(root, summary, output_dir)
        proposal = load_and_validate(proposal_path)
        experiment_path = materialize_experiment(
            root,
            proposal,
            output_dir,
            remaining_gpu_hours=remaining_gpu_hours,
        )
        print(f"Validated proposal: {proposal_path}")
        if not args.execute:
            print(f"Proposal-only mode; inspect {experiment_path} and rerun with --execute")
            break
        report_directory = (
            f"reports/generated/autonomous/iteration-{iteration:03d}-"
            f"{proposal['proposal_id']}"
        )
        flow_result = candidate_flow(
            str(experiment_path.relative_to(root)),
            report_directory,
        )
        summary = root / "artifacts" / "latest" / "summary.json"
        result = json.loads(summary.read_text(encoding="utf-8"))
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        neural_seconds = sum(
            float(run.get("metrics", {}).get("wall_clock_seconds", 0))
            for run in result.get("runs", [])
            if run.get("kind") == "neural"
        )
        gpu_hours_used += neural_seconds / 3600
        print(f"Completed bounded candidate cycle: {output_dir}")
        if flow_result["gate"]["passed"]:
            print("Stopped: the sealed evidence gate passed; promotion remains a human decision")
            break
    else:
        print("Stopped: maximum iteration count reached")


if __name__ == "__main__":
    main()
