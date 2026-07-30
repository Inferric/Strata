# Research decisions

| Decision | Rationale | Revisit trigger |
| --- | --- | --- |
| Start with `otbench` Mauna Loa | Public direct label and supported task API | Manifest/terms change or task cannot be reproduced |
| Chronological blocked split | Prevents adjacent-time leakage | Adopt a stricter official task split when exposed |
| Log-space probabilistic target | Cn² spans orders of magnitude and decisions need uncertainty | Observation model supports a better heavy-tail distribution |
| Surface model before profile model | Validates acquisition, controls, logging, and direct-label learning on 16 GB | Phase 1 gates pass |
| Teacher profile data kept separate | WRF/LES labels encode assumptions rather than truth | Never; separation is permanent |
| Lightning + Hydra + MLflow | Reusable training, composable config, evidence/artifact store | Measured friction exceeds benefit |
| Prefect rather than an opaque agent loop | Visible state, retry, schedule, and logs | Another engine demonstrably improves reliability |
| Optuna only after valid baseline | HPO cannot repair leakage or bad labels | Two-seed baseline and memory profile complete |
| DVC plus content manifests | Separate large bytes from Git while retaining identity | Data lake demands a catalog extension |
| LaTeX/PDF required | Scientific results need stable equations, tables, citations, and archival output | Never for completed scientific reports |
| Cloud disabled by default | Spending and data placement require specific approval | Approved launch packet |
| Human champion promotion | One autonomous scalar comparison is insufficient evidence | Never within the current governance model |
