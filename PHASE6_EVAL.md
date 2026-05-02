# Phase 6: Evaluation

This repo now includes a Phase 6 evaluator:

- `phase6_evaluate.py`

It runs the trained RL policy and/or baselines on the held-out test trace with:

- continuous, single-run test replay (no episode resets),
- deterministic RL inference (`predict(..., deterministic=True)` with action masks),
- frozen `VecNormalize` statistics loaded from training artifacts,
- per-step CSV logging and aggregate JSON/CSV summaries,
- optional stress scenarios and plots.

## Quick Start

Run full evaluation (RL + baselines):

```bash
python phase6_evaluate.py \
  --run-rl \
  --run-baselines \
  --models-root models/phase5 \
  --rl-experiments full \
  --make-plots

# Recommended stressed run for scheduler separation
python phase6_evaluate.py \
  --run-rl \
  --run-baselines \
  --models-root models/phase5 \
  --rl-experiments full \
  --scenarios high_load \
  --node-count 16 \
  --cpu-only-node-count 0 \
  --gpu-only-pods \
  --pod-replication-factor 4 \
  --high-load-mode reduce_per_node \
  --high-load-reduction-frac 0.0 \
  --make-plots

# Recommended fragmentation-focused run on a larger realistic cluster
python phase6_evaluate.py \
  --run-rl \
  --run-baselines \
  --models-root models/phase5 \
  --rl-experiments full \
  --scenarios full \
  --node-count 64 \
  --cpu-only-node-count 0 \
  --node-selection-policy stratified_gpu \
  --make-plots
```

Output folder (default):

- `evaluation/phase6/phase6_summary.json`
- `evaluation/phase6/phase6_results_table.csv`
- `evaluation/phase6/<scenario>/<method>_run*_steps.csv`
- plots (`*.png`) when `--make-plots` is enabled

## Stress Scenarios

Default scenarios:

- `full`
- `high_load`
- `gpu_intensive`
- `mixed`
- `heterogeneous`

Example:

```bash
python phase6_evaluate.py \
  --run-baselines \
  --scenarios full,high_load,gpu_intensive \
  --high-load-mode reduce_per_node \
  --high-load-reduction-frac 0.3 \
  --gpu-only-pods \
  --pod-replication-factor 4
```

You can force stronger resource pressure by reducing `--node-count` (for example `12-20`) and setting `--cpu-only-node-count 0`.

For a fragmentation-first comparison, keep the original test trace unreplicated (`--pod-replication-factor 1`, the default), sample 50-100 GPU nodes with `--node-selection-policy stratified_gpu`, and keep `--cpu-only-node-count 0` when you want to avoid CPU-only dilution.

## Baselines

Supported baseline names:

- `random`
- `best_fit`
- `dot_product`
- `gpu_packing`
- `gpu_clustering`
- `fgd`

Random baseline repeats are controlled by `--random-runs` (default `5`).

## RL Artifacts

Expected artifact layout (from Phase 5 training):

- `models/phase5/<experiment>/best_model.zip`
- `models/phase5/<experiment>/best_model.pkl` (or `best_vecnormalize.pkl`)

Where `<experiment>` is commonly `success_only`, `frag_only`, or `full`.

Important: RL checkpoints are tied to the observation shape used in training, which depends on `--node-count`.
If you evaluate with a different node count (for example training at 16 and evaluating at 64), RL loading will fail.
Use the same node count as training, or retrain for the new node count.

## Notes on GAR

The evaluator reports:

- `gar_capacity`: **time-averaged live GPU utilization ratio** (primary, bounded in `[0, 1]`)
- `gar_capacity_final`: live GPU utilization at end of trace
- `gar_capacity_cumulative_final`: cumulative scheduled GPU-milli divided by capacity (diagnostic; can exceed `1`)
- `cluster_full_free_gpu_count` / `final_full_free_gpu_count`: how many whole GPUs remain unused over time

This avoids misleading >1 values when traces are long.
