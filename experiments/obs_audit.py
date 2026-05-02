from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

from gpu_scheduling_env import build_env_from_preprocessed_csv, GPUSchedulingEnv


def collect_stats(env: GPUSchedulingEnv, max_obs: int = 10000) -> Dict[str, object]:
    obs_list: List[np.ndarray] = []
    strict_mask_zeros = 0
    safe_mask_zeros = 0
    steps = 0

    obs, info = env.reset()
    terminated = False
    while not terminated and steps < max_obs:
        mask = env.action_masks()
        strict = env.strict_action_mask()
        if np.all(strict == 0):
            strict_mask_zeros += 1
        if np.all(mask == 0):
            safe_mask_zeros += 1
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())

        # pick a valid action with the safe mask
        idxs = np.flatnonzero(mask)
        action = int(idxs[0]) if idxs.size else 0
        obs, reward, terminated, truncated, info = env.step(action)
        terminated = bool(terminated or truncated)
        steps += 1

    arr = np.asarray(obs_list, dtype=np.float32)
    dims = arr.shape[1] if arr.ndim == 2 else 0
    per_feature: Dict[str, List[float]] = defaultdict(list)
    stats: Dict[str, object] = {}
    if dims:
        cols = list(range(dims))
        mins = arr.min(axis=0).tolist()
        maxs = arr.max(axis=0).tolist()
        means = arr.mean(axis=0).tolist()
        stds = arr.std(axis=0).tolist()
        stats["dims"] = dims
        stats["n_obs"] = int(arr.shape[0])
        stats["per_feature_min"] = mins
        stats["per_feature_max"] = maxs
        stats["per_feature_mean"] = means
        stats["per_feature_std"] = stds

        # check against env.observation_space bounds
        low_arr = np.asarray(env.observation_space.low, dtype=np.float32)
        high_arr = np.asarray(env.observation_space.high, dtype=np.float32)
        stats["obs_space_low"] = low_arr.tolist()
        stats["obs_space_high"] = high_arr.tolist()
        # count features where any sample is out of bounds
        mask_oob = ((arr < low_arr) | (arr > high_arr)).any(axis=0)
        stats["features_out_of_bounds_count"] = int(mask_oob.sum())

    stats["strict_mask_all_zero_steps"] = int(strict_mask_zeros)
    stats["safe_mask_all_zero_steps"] = int(safe_mask_zeros)
    stats["steps"] = int(steps)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Observation & action-space audit for GPU scheduling env")
    parser.add_argument("--nodes-csv", type=Path, default=Path("preprocessed/phase3/nodes_clean.csv"))
    parser.add_argument("--pods-csv", type=Path, default=Path("preprocessed/phase3/pods_train.csv"))
    parser.add_argument("--demand-json", type=Path, default=Path("preprocessed/phase3/workload_demand_distribution.json"))
    parser.add_argument("--node-count", type=int, default=64)
    parser.add_argument("--episode-len", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-obs", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("experiments/obs_audit.json"))
    args = parser.parse_args()

    env = build_env_from_preprocessed_csv(
        nodes_csv=args.nodes_csv,
        pods_csv=args.pods_csv,
        demand_distribution_json=args.demand_json,
        node_count=args.node_count,
        max_pods_per_episode=args.episode_len,
        seed=args.seed,
    )

    stats = collect_stats(env, max_obs=args.max_obs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2))
    print(f"Wrote audit to {args.out}")


if __name__ == "__main__":
    main()
