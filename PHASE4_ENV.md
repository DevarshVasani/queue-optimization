# Phase 4/5: Environment and Training

This repository now includes a full Gym-compatible environment for RL scheduling and a Phase 5 training pipeline:

- `gpu_scheduling_env.py`: `GPUSchedulingEnv` (`gpu-scheduling-v0`)
- `test_gpu_scheduling_env.py`: smoke/invariant checks with masked random policy
- `train_maskable_ppo.py`: MaskablePPO training script (GPU-aware, validation GAR checkpointing, ablations)
- `phase6_evaluate.py`: Phase 6 evaluation runner (RL vs baselines, stress tests, plots)

## 1) Environment design

The environment follows one scheduling decision per step:

1. Advance simulator time to the next arrival or completion that can produce a feasible pending pod.
2. Process completions via simulator internal state updates.
3. Keep infeasible pods pending instead of treating them as immediately failed.
4. Build observation vector.
5. Apply action (node index), schedule pod if feasible.
6. Compute a latency-objective reward and move to the next feasible pending pod.

It uses `ClusterSimulator` from `cluster_sim.py` and new API methods added there.

## 2) Observation/action spaces

- Action space: `Discrete(N)` where `N=node_count`
- Observation: flat vector
  - pod block: 17 dims
  - node block: `N * 8` dims
  - global block: 10 dims
  - total: `17 + 7N + 10`

For `N=128`, shape is `1051`.

The environment exposes:

- `action_masks()` -> safe mask (all ones fallback when no feasible action exists)
- `strict_action_mask()` -> strict feasibility mask (all zeros possible)

Use `strict_action_mask` in diagnostics, and `action_masks` for MaskablePPO compatibility.

## 3) Reward objective

Default training uses `reward_mode=latency`, a weight-free objective aligned with job completion time and tail latency:

- Each scheduled job records JCT as `scheduled_time + runtime - creation_time`.
- JCT is normalized by runtime to get slowdown, so short and long jobs are comparable without tuned constants.
- The episode objective is the RMS of mean, P95, and P99 slowdown.
- The step reward is the negative change in that objective, so the cumulative reward is the negative final latency objective.

The older shaped placement reward is still available with `reward_mode=legacy` for ablations, but it is not the default research objective.

## 4) Phase 5 training support

Install dependencies:

```bash
pip install -r requirements-rl.txt
```

Train full reward (auto chooses CUDA when available):

```bash
python train_maskable_ppo.py --device auto
```

Train from config file (CLI flags still override config values):

```bash
python train_maskable_ppo.py --config configs/phase5_maskable_ppo.json --device auto
```

Force GPU:

```bash
python train_maskable_ppo.py --device cuda
```

Run ablations in one command:

```bash
python train_maskable_ppo.py --ablations success_only,frag_only,full --device auto
```

The script includes:

- `MaskablePPO` with `MlpPolicy` and net `[256, 256]`
- `VecNormalize(norm_obs=True, norm_reward=True)` for training
- custom validation callback selecting best checkpoint by the validation latency objective
- validation logging for mean/P95/P99 JCT, mean/P95/P99 slowdown, fragmentation, utilization spread, and wait time
- early stopping by `--patience-steps`
- saved artifacts per experiment in `models/phase5/<ablation>/`
  - `best_model.zip` (policy checkpoint)
  - `best_model.pkl` and `best_vecnormalize.pkl` (normalization stats)

## 5) Environment smoke test

Run quick validation:

```bash
python test_gpu_scheduling_env.py --node-count 20 --episode-len 100 --episodes 3
```

Checks include:

- observation shape consistency
- mask-driven action validity
- step/reset loop correctness

## 6) Notes

- Phase 4 uses a fixed `node_count` subset for tractable MLP training.
- Episodes are fixed-length, non-overlapping windows.
- Sub-placement policy defaults to `most_used_first`.
- Validation in training uses long validation episodes (`--val-episode-len 0` => full validation split).
- This environment tracks per-episode GAR/success/fragmentation plus mean/P95/P99 JCT and slowdown via `info["episode_metrics"]`.
