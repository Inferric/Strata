Continue Strata-OT as a long-running autonomous neural-architecture research program. The completed MLO weather-horizon experiment is Cycle 0, not the end of the project. Its result established that operational weather contains useful signal for LightGBM, while Strata-OT Horizon v1 failed to use it reliably. Preserve that negative result and continue.

Read AGENTS.md, the research documents, experiment ledger, PR #2, the Cycle 0 reports, and use $cn2-research-pipeline plus relevant project skills.

Operate autonomously for up to 12 wall-clock hours and 8 local GPU-hours, targeting at least 10 hours of productive work. Do not stop merely because one experiment completes or fails. After every cycle, analyze the evidence, record the result, preregister the next justified hypothesis, validate its run contract, and execute the next cycle. Never idle solely to satisfy the duration. If one hypothesis family produces three consecutive non-improvements, close that family and pivot to another justified architecture, data, calibration, or robustness hypothesis instead of ending the entire program.

Treat LightGBM strictly as a strong diagnostic baseline, never as the intended flagship model. The objective is to develop our own neural model family and determine what it needs to outperform strong controls honestly.

First repair the experiment machinery:
- Replace completed gate states of OPEN with PASS, FAIL, or NOT_EVALUATED.
- Correct GPU-memory and artifact-storage accounting.
- Add tests preventing an evaluated failed condition from remaining OPEN.
- Mark the previously viewed development assessment block as consumed. Never tune against it.
- Keep the official MLO test partition sealed.

Create a new rolling-origin chronological development protocol with purged boundaries and multiple weather periods. Fit preprocessing and calibration only on the appropriate training folds. Use selection folds for iteration; reserve a new untouched confirmation block for the final frozen candidate.

Analyze LightGBM only to discover which weather variables, lags, derivatives, and interactions contain signal. Do not copy or ship the tree model as Strata-OT.

Design and implement Strata-OT Fusion v2 as a custom neural architecture with:
- short-, medium-, and slow-timescale Cn² context;
- persistence-anchored residual prediction;
- horizon-conditioned 5/15/30/60-minute decoding;
- weather conditioning throughout the temporal representation using cross-attention, feature-wise modulation, or another justified interaction mechanism rather than one suppressible final scalar gate;
- physically meaningful derived weather/stability/refractivity tokens where supported by available public data;
- probabilistic quantiles, heavy-tail-aware uncertainty, and OOD diagnostics;
- interpretable logged contributions without making causal regime claims;
- a local target of roughly 2–8M parameters suitable for the 16GB RTX 5080.

Implement atmospheric multi-task pretraining using available public training data: masked reconstruction and future prediction of weather variables, cadence-aware Cn² prediction, and any scientifically defensible auxiliary tasks. Do not use assessment or test labels during pretraining.

Run a structured ablation program covering:
1. context length and timescale branches;
2. weather-fusion mechanism;
3. physics-derived tokens;
4. multi-task pretraining;
5. model capacity and regularization;
6. probabilistic calibration.

Use cheap selection-only screens first. Advance only promising candidates to multiple rolling folds and at least three seeds. Always compare against persistence, climatology, MLP/TCN, LightGBM, Horizon v1, and the current best Strata candidate. Do not weaken frozen metrics or promotion thresholds.

After each cycle, log configs, code/data/split identities, checkpoints, predictions, metrics, uncertainty, peak total VRAM, runtime, failures, and plots to MLflow. Update the actual research console and maintain a chronological experiment ledger. Generate a compiled LaTeX report for every substantial experiment and a final program-level synthesis.

If neural performance remains data-limited, autonomously investigate and acquire additional clearly open public optical-turbulence or meteorological datasets through provenance-checked adapters. Skip ambiguous licenses or gated terms without ending the rest of the program. Begin scaffolding Strata-OT Column only if the necessary public profile data passes provenance and split checks.

Do not access the official MLO test, promote a champion, merge the PR, rent cloud GPUs, spend money, accept terms, expose credentials, or interact with controlled data without human approval. Work only on public atmospheric-science modeling.

Maintain checkpoint commits and update the existing draft PR #2, but do not merge it.

The program is complete only when:
- the gate/accounting defects are fixed and tested;
- a new leakage-resistant rolling protocol exists;
- Fusion v2 and multi-task pretraining are implemented;
- the complete ablation program has run;
- the best custom candidate has at least three-seed, multi-fold evidence;
- its relationship to LightGBM is stated honestly;
- all artifacts are visible in MLflow and the frontend;
- tests, Ruff, mypy, frontend build, manifests, splits, and LaTeX compilation pass;
- a visually inspected final PDF explains every experiment, negative result, best architecture, limitations, and next research phase.

Do not declare completion after an intermediate null result. Stop early only for a genuine safety, provenance, hardware, permissions, credential, or human-approval blocker. Preserve all completed work and describe the exact blocker if one occurs.