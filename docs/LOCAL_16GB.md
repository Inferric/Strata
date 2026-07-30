# RTX 5080 / 16 GB operating profile

The first run is deliberately small. BF16, Flash/SDPA attention supplied by
PyTorch, bounded context, activation-conscious widths, and gradient accumulation
are enough for the 2–8M parameter Surface model and the first 8–15M Column
model.

## Order of operations

1. Verify a current NVIDIA driver and the PyTorch CUDA 12.8 wheel.
2. Run one batch on CPU for shape/schema checks.
3. Run `--fast-dev-run` on GPU.
4. Record peak VRAM.
5. Increase batch size only while staying below 15.5 GB.
6. Complete seed 17, then seed 41 if stable and inside four GPU-hours.

## OOM recovery

One automatic recovery is permitted:

- halve batch size;
- double gradient accumulation;
- keep the effective batch approximately stable; and
- record both the failure and recovered configuration.

Do not silently shorten the data, context, model, or evaluation set after seeing
results. Persistent OOM is a stop condition.

## Why not start at 20–120M parameters

The full briefing identified that as a credible later profile publication
range, not the correct first experiment. On a 16 GB card, data validity,
split quality, calibration, and direct-label evidence have much higher expected
value than maximizing parameter count.
