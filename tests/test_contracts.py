from __future__ import annotations

from pathlib import Path

import pytest

from strata_ot.autonomy.contracts import validate_proposal
from strata_ot.tools.validate_repo import validate_repository


def valid_proposal() -> dict[str, object]:
    return {
        "proposal_id": "surface-depth-ablation-v1",
        "hypothesis": (
            "Reducing depth while preserving width will retain blocked-time error "
            "and reduce peak memory on the local device."
        ),
        "parent_run_id": None,
        "dataset_manifest_id": "otbench-mlo-cn2-15m-v1",
        "split_id": "otbench-mlo-blocked-v1",
        "model": {
            "family": "strata_ot_surface",
            "config_path": "configs/model/strata_surface_tiny.yaml",
        },
        "changes": [
            {"parameter": "depth", "old": 6, "new": 4, "reason": "Test depth efficiency"}
        ],
        "evaluation_gate_id": "public-evidence-gates-v1",
        "budget": {
            "max_runs": 2,
            "max_local_gpu_hours": 4,
            "max_cloud_cost_usd": 0,
        },
        "stop_conditions": [
            "budget_exhausted",
            "nan_or_divergence",
            "oom_after_one_recovery",
            "data_or_split_gate_failure",
        ],
    }


def test_valid_proposal() -> None:
    validate_proposal(valid_proposal())


def test_proposal_rejects_cloud_cost() -> None:
    proposal = valid_proposal()
    proposal["budget"]["max_cloud_cost_usd"] = 1  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_proposal(proposal)


def test_proposal_rejects_unbounded_or_stale_change() -> None:
    proposal = valid_proposal()
    proposal["changes"] = [
        {
            "parameter": "hidden_dim",
            "old": 999,
            "new": 1024,
            "reason": "Attempt an oversized stale configuration",
        }
    ]
    with pytest.raises(ValueError):
        validate_proposal(proposal)


def test_proposal_requires_hard_stops() -> None:
    proposal = valid_proposal()
    proposal["stop_conditions"] = [
        "budget_exhausted",
        "nan_or_divergence",
        "no_validation_improvement",
    ]
    with pytest.raises(ValueError):
        validate_proposal(proposal)


def test_fusion_proposal_allows_bounded_training_sequence_scale() -> None:
    proposal = valid_proposal()
    proposal.update(
        {
            "proposal_id": "fusion-v22-data-scale-030k",
            "dataset_manifest_id": "otbench-usna-cn2-lg-v1",
            "split_id": "otbench-usna-lg-fusion-v2",
            "model": {
                "family": "strata_ot_fusion",
                "config_path": "configs/model/strata_fusion_v2.yaml",
            },
            "changes": [
                {
                    "parameter": "max_train_examples",
                    "old": 12000,
                    "new": 30000,
                    "reason": "Increase only training density in the frozen fold",
                }
            ],
        }
    )
    validate_proposal(proposal)


def test_fusion_proposal_allows_bounded_screen_epoch_cap() -> None:
    proposal = valid_proposal()
    proposal.update(
        {
            "proposal_id": "fusion-v25-convergence-9",
            "dataset_manifest_id": "otbench-usna-cn2-lg-v1",
            "split_id": "otbench-usna-lg-fusion-v2",
            "model": {
                "family": "strata_ot_fusion",
                "config_path": "configs/model/strata_fusion_v2.yaml",
            },
            "changes": [
                {
                    "parameter": "screen_max_epochs",
                    "old": 3,
                    "new": 9,
                    "reason": "Test fixed-architecture convergence",
                }
            ],
        }
    )
    validate_proposal(proposal)


def test_fusion_proposal_allows_bounded_full_epoch_cap() -> None:
    proposal = valid_proposal()
    proposal.update(
        {
            "proposal_id": "fusion-v25-rolling-robustness",
            "dataset_manifest_id": "otbench-usna-cn2-lg-v1",
            "split_id": "otbench-usna-lg-fusion-v2",
            "model": {
                "family": "strata_ot_fusion",
                "config_path": "configs/model/strata_fusion_v2.yaml",
            },
            "changes": [
                {
                    "parameter": "full_max_epochs",
                    "old": 12,
                    "new": 9,
                    "reason": "Preserve the selected convergence cap",
                }
            ],
        }
    )
    validate_proposal(proposal)


def test_repository_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    assert validate_repository(root) == []
