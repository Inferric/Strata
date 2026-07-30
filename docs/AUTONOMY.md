# Semi-autonomous research loop

Codex is the hypothesis/coding layer, not the source of record. Prefect executes
visible flows; MLflow stores run evidence; DVC and manifests identify data;
schemas and sealed gates bound what can change.

## Loop

1. Read the latest completed summary, residual analysis, failures, and budget.
2. Ask `codex exec` for one JSON proposal using
   `schemas/experiment_proposal.schema.json`.
3. Validate the schema, dataset/split identity, allowlisted parameters, run
   count, local GPU-hours, cloud dollars, and stop conditions.
4. Materialize a generated config without editing the parent config.
5. Run through Prefect and log to MLflow.
6. Evaluate sealed gates and compile the LaTeX report.
7. Stop for review or request another bounded proposal.

`strata-loop` is proposal-only unless `--execute` is passed. A caller may set
`--max-iterations`, but all cycles share the explicit
`--max-total-gpu-hours` ceiling (hard-capped at four local hours). Every
executed cycle runs through Prefect, writes MLflow evidence, evaluates the
sealed gate, and compiles a LaTeX/PDF report. The controller stops when the gate
passes, the shared budget expires, or the iteration limit is reached. It cannot
assign cloud dollars, edit sealed gates, accept data terms, or promote a
champion.

## Hard stops

- credentials, login, click-through, or license acceptance;
- an upstream manifest or checksum change;
- any random-row primary evaluation proposal;
- preprocessing leakage or group/time overlap;
- NaN/divergence;
- OOM after one conservative batch/accumulation recovery;
- exhausted GPU-hour/run/dollar budget;
- cloud rental without explicit approval;
- any public-release or controlled-data question;
- model-registry champion promotion.

## Codex invocation

The controller uses non-interactive `codex exec` with a read-only sandbox,
ephemeral session, JSON event log, output schema, and captured final proposal.
It does not load `.env` or expose credentials to repository lifecycle hooks.
Scheduled execution is still bounded by the same schema and authorization; a
schedule is not broader permission.

## What “continuous” means here

Continuous means resumable, observable, and capable of proposing the next
experiment while a fixed budget remains. It does not mean an infinite sweep.
Repeated failure, weak validation gain, or exhausted evidence ends the loop.
