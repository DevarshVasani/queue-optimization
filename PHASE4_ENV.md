# Phase 4/5: Environment and Training

This repository now includes a full Gym-compatible environment for RL scheduling and a Phase 5 training pipeline:

- `gpu_scheduling_env.py`: `GPUSchedulingEnv` (`gpu-scheduling-v0`)
- `test_gpu_scheduling_env.py`: smoke/invariant checks with masked random policy
- `train_maskable_ppo.py`: MaskablePPO training script (GPU-aware, validation GAR checkpointing, ablations)
- `phase6_evaluate.py`: Phase 6 evaluation runner (RL vs baselines, stress tests, plots)

## 1) Environment design

The environment follows one decision per step:

1. Advance simulator time to current pod arrival.
2. Process completions via simulator internal state updates.
3. Compute strict feasibility mask for all nodes.
4. Build observation vector.
5. Apply action (node index), schedule pod if feasible.
6. Compute reward components and move to next pod.

It uses `ClusterSimulator` from `cluster_sim.py` and new API methods added there.

## 2) Observation/action spaces

- Action space: `Discrete(N)` where `N=node_count`
- Observation: flat vector
  - pod block: 17 dims
  - node block: `N * 7` dims
  - global block: 10 dims
  - total: `17 + 7N + 10`

For `N=128`, shape is `923`.

The environment exposes:

- `action_masks()` -> safe mask (all ones fallback when no feasible action exists)
- `strict_action_mask()` -> strict feasibility mask (all zeros possible)

Use `strict_action_mask` in diagnostics, and `action_masks` for MaskablePPO compatibility.

## 3) Reward components

`R = R_succ + R_frag + R_bal + R_slo`

- `R_succ`: `success_reward` or `fail_penalty`
- `R_frag`: `-frag_weight * (F_after - F_before)` for chosen node
- `R_bal`: `-balance_weight * (sigma_cpu + sigma_mem + sigma_gpu_util)`
- `R_slo`: optional penalty when wait exceeds threshold (scaled for high priority)

Defaults:

- `frag_weight=2.0`
- `balance_weight=0.5`
- `success_reward=1.0`
- `fail_penalty=-5.0`
- `slo_penalty=-2.0`
- `slo_threshold_ms=30000`

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
- custom validation callback selecting best checkpoint by validation `GAR` (`allocated/requested`)
- validation logging for fragmentation, utilization spread (`sigma_cpu`, `sigma_mem`, `sigma_gpu`), and wait time
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
- This environment tracks per-episode GAR/success/fragmentation metrics via `info["episode_metrics"]`.
