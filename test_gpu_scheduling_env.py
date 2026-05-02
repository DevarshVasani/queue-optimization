from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gpu_scheduling_env import build_env_from_preprocessed_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test GPU scheduling Gym environment")
    parser.add_argument(
        "--nodes-csv",
        type=Path,
        default=Path("preprocessed/phase3/nodes_clean.csv"),
    )
    parser.add_argument(
        "--pods-csv",
        type=Path,
        default=Path("preprocessed/phase3/pods_train.csv"),
    )
    parser.add_argument(
        "--demand-json",
        type=Path,
        default=Path("preprocessed/phase3/workload_demand_distribution.json"),
    )
    parser.add_argument("--node-count", type=int, default=20)
    parser.add_argument("--episode-len", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = build_env_from_preprocessed_csv(
        nodes_csv=args.nodes_csv,
        pods_csv=args.pods_csv,
        demand_distribution_json=args.demand_json,
        node_count=args.node_count,
        max_pods_per_episode=args.episode_len,
        seed=args.seed,
    )

    print(f"obs_dim={env.observation_space.shape[0]} action_n={env.action_space.n}")
    total_reward = 0.0
    scheduled = 0
    unscheduled = 0

    for ep in range(args.episodes):
        obs, info = env.reset()
        if obs.shape != env.observation_space.shape:
            raise RuntimeError(f"observation shape mismatch: {obs.shape} != {env.observation_space.shape}")

        done = False
        ep_reward = 0.0
        step_idx = 0
        while not done:
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                action = 0
            else:
                action = int(np.random.default_rng(args.seed + ep + step_idx).choice(valid))

            obs, reward, terminated, truncated, step_info = env.step(action)
            strict_mask = step_info["strict_action_mask"]
            if np.any(strict_mask) and strict_mask[action] == 0:
                raise RuntimeError("selected infeasible action despite non-empty strict mask")

            ep_reward += float(reward)
            scheduled += int(step_info["scheduled"])
            unscheduled += 0 if step_info["scheduled"] else 1
            done = bool(terminated or truncated)
            step_idx += 1

        episode_metrics = step_info.get("episode_metrics", {})
        if episode_metrics:
            gar_capacity = float(episode_metrics.get("gar", 0.0))
            gar_requested = float(episode_metrics.get("gar_requested", 0.0))
            unalloc = float(episode_metrics.get("unallocated_gpu_fraction", 0.0))
            mean_jct = float(episode_metrics.get("mean_job_completion_time_ms", 0.0))
            p95_jct = float(episode_metrics.get("p95_job_completion_time_ms", 0.0))
            p99_jct = float(episode_metrics.get("p99_job_completion_time_ms", 0.0))
            latency_obj = float(episode_metrics.get("latency_objective", 0.0))
            print(
                f"episode={ep} GAR_capacity={gar_capacity:.4f} "
                f"GAR_requested={gar_requested:.4f} unallocated={unalloc:.4f} "
                f"mean_JCT_ms={mean_jct:.1f} p95_JCT_ms={p95_jct:.1f} "
                f"p99_JCT_ms={p99_jct:.1f} latency_obj={latency_obj:.4f}"
            )

        total_reward += ep_reward
        print(
            f"episode={ep} steps={step_idx} reward={ep_reward:.4f} "
            f"scheduled={scheduled} unscheduled={unscheduled}"
        )

    print(
        f"completed episodes={args.episodes} total_reward={total_reward:.4f} "
        f"scheduled={scheduled} unscheduled={unscheduled}"
    )


if __name__ == "__main__":
    main()
