# Cloud GPU scale-up

Cloud execution is optional and disabled by default. Provider price and
availability change too quickly to hardcode; query them at launch time and
record the quoted and realized cost in MLflow.

## Adapter

SkyPilot is the portable job layer. Provider-native CLIs may be used for
diagnostics, but experiments should retain the same container/environment,
configuration, checkpoint, manifest, and reporting contract across local and
cloud runs.

The initial cloud target is one 48–80 GB GPU for the Column model or a bounded
Optuna study. Multi-node training is not justified until one-node scaling and
data throughput are measured.

## Approval packet

Before a launch, produce:

- hypothesis and expected information gain;
- exact provider/region/GPU and live hourly quote;
- maximum dollars, wall time, and job count;
- on-demand versus spot tradeoff;
- checkpoint/recovery and automatic teardown tests;
- data jurisdiction/terms review;
- secret scopes; and
- local reduced-config reproduction command.

## Cost control

The research controller requires `STRATA_ALLOW_CLOUD=true` and a positive
`STRATA_MAX_CLOUD_COST_USD`, but those environment values are not themselves
authorization. A human must approve the specific launch. Terminate resources on
completion/failure and reconcile actual spend into the run report.
