from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import jsonschema

from strata_ot.config import find_repo_root, load_yaml

ALLOWED_MODEL_PARAMETERS = {
    "hidden_dim",
    "depth",
    "num_heads",
    "num_experts",
    "dropout",
    "context",
    "learning_rate",
    "weight_decay",
    "scale_parameterization",
    "mlp_flatten_context",
    "fusion",
    "active_scales",
    "physics_tokens",
    "pretraining",
    "calibration",
    "point_loss_weight",
    "max_train_examples",
    "residual_horizon_exponent",
    "residual_shortcut",
}

REQUIRED_STOP_CONDITIONS = {
    "budget_exhausted",
    "nan_or_divergence",
    "oom_after_one_recovery",
    "data_or_split_gate_failure",
}

EXPECTED_DATASET_SPLITS = {
    "otbench-mlo-cn2-15m-v1": "otbench-mlo-blocked-v1",
    "otbench-mlo-cn2-15m-weather-horizon-v1": (
        "otbench-mlo-weather-horizon-v1"
    ),
    "otbench-usna-cn2-3m-forecast-6min-v1": "otbench-usna-sm-forecast-v1",
    "otbench-usna-cn2-lg-v1": "otbench-usna-lg-fusion-v2",
}

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "hidden_dim": (64, 384),
    "depth": (2, 10),
    "num_heads": (2, 12),
    "num_experts": (2, 8),
    "dropout": (0.0, 0.3),
    "context": (6, 128),
    "learning_rate": (1e-5, 3e-3),
    "weight_decay": (0.0, 0.2),
    "point_loss_weight": (0.0, 2.0),
    "max_train_examples": (1000, 500000),
    "residual_horizon_exponent": (0.0, 1.5),
}

INTEGER_PARAMETERS = {
    "hidden_dim",
    "depth",
    "num_heads",
    "num_experts",
    "context",
    "max_train_examples",
}


def validate_proposal(proposal: dict[str, Any], root: Path | None = None) -> None:
    repository = root or find_repo_root()
    schema = json.loads(
        (repository / "schemas" / "experiment_proposal.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(proposal)
    if float(proposal["budget"]["max_cloud_cost_usd"]) != 0:
        raise ValueError("Autonomous local proposals must have a zero cloud budget")
    if int(proposal["budget"]["max_runs"]) > 2:
        raise ValueError("Autonomous proposals may request at most two runs")
    dataset_id = str(proposal["dataset_manifest_id"])
    if dataset_id not in EXPECTED_DATASET_SPLITS:
        raise ValueError("Proposal changed the approved dataset identity")
    if proposal["split_id"] != EXPECTED_DATASET_SPLITS[dataset_id]:
        raise ValueError("Proposal changed the frozen split identity")
    missing_stops = REQUIRED_STOP_CONDITIONS - set(proposal["stop_conditions"])
    if missing_stops:
        raise ValueError(
            "Proposal omitted required stop conditions: " + ", ".join(sorted(missing_stops))
        )

    config_path = (repository / str(proposal["model"]["config_path"])).resolve()
    approved_model_dir = (repository / "configs" / "model").resolve()
    if not config_path.is_relative_to(approved_model_dir) or not config_path.is_file():
        raise ValueError("Proposal model config is outside the approved model directory")
    model_config = load_yaml(config_path, root=repository)
    parent_reference = proposal["model"].get("parent_config_path")
    parent_config = (
        load_yaml(str(parent_reference), root=repository)
        if parent_reference is not None
        else model_config
    )
    trainer_config = load_yaml("configs/trainer/local_16gb.yaml", root=repository)
    fusion_trainer_config = load_yaml(
        "configs/trainer/fusion_local_16gb.yaml",
        root=repository,
    )

    changed: set[str] = set()
    for change in proposal["changes"]:
        parameter = str(change["parameter"])
        if parameter not in ALLOWED_MODEL_PARAMETERS:
            raise ValueError(f"Autonomous change is not allowlisted: {parameter}")
        if parameter in changed:
            raise ValueError(f"Autonomous proposal changes {parameter} more than once")
        changed.add(parameter)
        if parameter == "max_train_examples":
            source = {
                "max_train_examples": fusion_trainer_config[
                    "screen_max_train_examples"
                ]
            }
        else:
            source = (
                trainer_config
                if parameter in {"learning_rate", "weight_decay"}
                else parent_config
            )
        defaults: dict[str, Any] = {
            "scale_parameterization": "clamp",
            "mlp_flatten_context": False,
            "point_loss_weight": 0.0,
        }
        old_value = source.get(parameter, defaults.get(parameter))
        if old_value is None or change["old"] != old_value:
            raise ValueError(f"Proposal old value does not match the sealed parent for {parameter}")
        new_value = change["new"]
        if parameter == "scale_parameterization":
            if new_value != "softplus":
                raise ValueError("Forecast uncertainty may only change to softplus")
            continue
        if parameter == "mlp_flatten_context":
            if new_value is not True:
                raise ValueError("Forecast MLP must consume the complete context")
            continue
        categorical: dict[str, set[Any]] = {
            "fusion": {"concat", "film", "film_cross_attention"},
            "active_scales": {
                ("short",),
                ("short", "medium"),
                ("short", "medium", "slow"),
            },
            "physics_tokens": {"none", "base", "full"},
            "pretraining": {"none", "reconstruction", "full_multitask"},
            "calibration": {"none", "conformal", "student_t_conformal"},
            "residual_shortcut": {
                "none",
                "history_linear",
                "all_linear",
                "all_mlp",
            },
        }
        if parameter in categorical:
            comparable = (
                tuple(str(value) for value in new_value)
                if parameter == "active_scales" and isinstance(new_value, list)
                else new_value
            )
            if comparable not in categorical[parameter]:
                raise ValueError(
                    f"Autonomous categorical value is invalid for {parameter}"
                )
            continue
        if isinstance(new_value, bool) or not isinstance(new_value, (int, float)):
            raise ValueError(f"Autonomous value for {parameter} must be numeric")
        if parameter in INTEGER_PARAMETERS and not isinstance(new_value, int):
            raise ValueError(f"Autonomous value for {parameter} must be an integer")
        lower, upper = NUMERIC_BOUNDS[parameter]
        numeric = float(new_value)
        if not math.isfinite(numeric) or not lower <= numeric <= upper:
            raise ValueError(
                f"Autonomous value for {parameter} must be between {lower} and {upper}"
            )
        if new_value == change["old"]:
            raise ValueError(f"Autonomous change for {parameter} is a no-op")
        if (
            parent_reference is not None
            and parameter in model_config
            and model_config[parameter] != new_value
        ):
            raise ValueError(
                f"Proposal final value does not match {config_path.name} "
                f"for {parameter}"
            )

    resolved_model = dict(parent_config)
    for change in proposal["changes"]:
        if change["parameter"] not in {"learning_rate", "weight_decay"}:
            resolved_model[str(change["parameter"])] = change["new"]
    if int(resolved_model["hidden_dim"]) % int(resolved_model["num_heads"]) != 0:
        raise ValueError("hidden_dim must remain divisible by num_heads")


def load_and_validate(path: Path) -> dict[str, Any]:
    proposal = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    validate_proposal(proposal)
    return proposal
