# Research console contract

The React console is a local evidence interface, not a marketing website.

It reads only recorded state:

- recent MLflow runs and model families;
- blocked-test metrics, uncertainty, VRAM, and time;
- dataset manifests and label provenance;
- evidence-gate status;
- Prefect/MLflow/API health;
- local/cloud budget state; and
- links to native MLflow and Prefect views.

If services or runs are absent, it shows an explicit empty state. Demo or
synthetic metrics are prohibited. MLflow remains the artifact/model source of
record; Prefect remains the workflow/log source of record; the API is an
aggregation layer.

Planned additions after real data exist:

- residual and calibration plots;
- vertical profile overlays with uncertainty;
- regime-router diagnostics;
- dataset coverage maps/timelines;
- cost and peak-memory traces; and
- candidate-versus-champion comparison.
