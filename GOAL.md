# First autonomous Codex goal

Copy the text below into `/goal` from the repository root.

```text
Bring this repository from a clean checkout to its first reproducible real Cn² training result.

Read AGENTS.md and use $cn2-research-pipeline plus the relevant project skills. Work only with public, openly downloadable data. Install and validate the local toolchain, start the PostgreSQL, MLflow, Prefect, API, and research-console services, and verify their health.

Acquire the public otbench Mauna Loa Cn² task through its supported interface. Record provenance and create a frozen blocked-time split without changing any sealed evaluation gate. Train and evaluate:
1. persistence/climatology where applicable,
2. a compact MLP neural baseline,
3. Strata-OT Surface using the local_16gb configuration.

Use the RTX 5080 as a 16 GB device. Begin with fast-dev and memory-probe runs, then complete at least two seeds for the best neural configuration if the first run is stable. Keep the total local GPU budget under four hours. Do not rent cloud resources or accept gated dataset terms.

Log every run, resolved config, dataset/split identity, metrics, peak VRAM, checkpoints, and figures to MLflow. Run the repository evaluation gates. Make the API and frontend display the actual completed runs rather than demo data.

Generate a LaTeX experiment report and compile it to PDF. The report must state the hypothesis, data coverage, provenance, split, models, exact configuration, metrics, uncertainty across available seeds, failures, limitations, and recommended next experiment.

Done means: tests pass; services are healthy; the real runs and model artifacts are visible in MLflow and the research console; no leakage/provenance gate fails; and the compiled PDF exists under reports/generated/first-real-run/.

If installation, data access, CUDA compatibility, or hardware blocks progress, diagnose and document the exact blocker, preserve all completed reproducible work, and stop rather than substituting synthetic results.
```
