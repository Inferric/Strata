# Autonomy and reporting

## Control loop

1. Read the immutable benchmark state and champion summary.
2. Generate one structured proposal containing a hypothesis, allowed changes, data/split IDs, model/config overrides, seed set, resource ceiling, cost ceiling, expected signal, and stop rule.
3. Validate the proposal before execution.
4. Execute in an isolated branch or worktree.
5. Log metrics and artifacts; run all required gates.
6. Compile a LaTeX experiment report.
7. Accept, reject, or request human review.
8. Continue only when the remaining budget and evidence justify another experiment.

## Codex boundary

Prefer non-interactive `codex exec` with JSON output constrained by a schema. Codex may propose or edit:

- model components
- training/evaluation code
- experiment configuration
- tests
- analysis and LaTeX report source

Codex must not autonomously change:

- raw data
- frozen split files
- benchmark labels
- promotion thresholds
- secret/credential files
- public-release classifications
- cloud spending or resource ceilings

Keep API credentials scoped to the single command that needs them. Do not expose them to repository-controlled setup hooks.

## Stop conditions

Stop immediately on:

- missing or contradictory provenance
- access terms requiring a person
- NaN/Inf loss after one safe retry
- GPU OOM after bounded batch/accumulation adjustment
- thermal, disk, or quota alarm
- estimated spend above the approved ceiling
- three consecutive cycles without a meaningful improvement
- evidence of leakage, corrupted labels, or sealed-test access
- a request to operationalize weapon effects or publish controlled integration details

## LaTeX report contract

Every completed run must produce source and PDF containing:

- title, run ID, timestamp, code/data/split IDs, environment and hardware
- hypothesis and preregistered win condition
- data coverage and QC
- model and resolved hyperparameters
- metrics with uncertainty across seeds
- figures and failure analysis
- comparison with baselines/champion
- limitations, negative results, and next experiment
- claim-to-evidence table

Keep figures as vector PDF where practical. Compile and inspect the PDF before declaring completion.
