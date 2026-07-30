from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strata_ot.data.acquire import _canonical_target_frame


class FakeTask:
    def __init__(self, raw: pd.DataFrame):
        self.raw = raw

    def get_transforms(self) -> dict[str, dict[str, bool]]:
        return {"log_transform": {"value": True}}

    def get_df(self) -> pd.DataFrame:
        return self.raw

    def get_target_name(self) -> str:
        return "Cn2_15m"


def test_canonical_target_is_fixed_precision_and_aligned() -> None:
    index = pd.to_datetime(["2006-06-09T00:00:00", "2006-06-09T00:05:00"])
    raw = pd.DataFrame(
        {"Cn2_15m": np.asarray([1.2345678e-13, 9.876543e-15], dtype=np.float32)},
        index=index,
    )
    upstream = pd.DataFrame(
        {"Cn2_15m": np.log10(raw["Cn2_15m"])},
        index=index,
    )

    canonical = _canonical_target_frame(FakeTask(raw), upstream)

    assert canonical["_upstream_index"].tolist() == list(index)
    assert all(value.count(".") == 1 for value in canonical["target"])
    assert all(len(value.rsplit(".", maxsplit=1)[1]) == 12 for value in canonical["target"])
    assert np.allclose(
        canonical["target"].astype(float),
        upstream["Cn2_15m"].to_numpy(float),
        rtol=0,
        atol=2e-6,
    )


def test_canonical_target_rejects_material_task_difference() -> None:
    index = pd.to_datetime(["2006-06-09T00:00:00"])
    raw = pd.DataFrame({"Cn2_15m": np.asarray([1e-13], dtype=np.float32)}, index=index)
    inconsistent = pd.DataFrame({"Cn2_15m": [-12.0]}, index=index)

    with pytest.raises(RuntimeError, match="differs materially"):
        _canonical_target_frame(FakeTask(raw), inconsistent)
