from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(gt=0)


class DatasetTerms(BaseModel):
    status: Literal["open", "review_required", "gated"]
    redistribution: Literal["allowed", "metadata_only", "unknown"]


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    dataset_id: str
    source_url: str
    citation: str
    accessed_at: datetime
    terms: DatasetTerms
    files: list[DatasetFile]
    label_provenance: Literal[
        "direct_observation",
        "numerical_weather_model_teacher",
        "large_eddy_simulation_teacher",
        "derived_proxy",
        "mixed",
    ]
    geometry: dict[str, Any]
    instrument: dict[str, Any] | None = None
    wavelength_nm: float | None = None
    units: dict[str, Any]
    coverage: dict[str, Any]
    qc: dict[str, Any]
