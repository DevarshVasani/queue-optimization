from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from cluster_sim import Node, GPU, Pod
from gpu_scheduling_env import GPUSchedulingEnv


def make_minimal_cluster() -> tuple[list[Node], list[Pod]]:
    nodes = [
        Node(node_id="n0", cpu_total=2000, memory_total=4096, model="T4", gpu_list=[GPU(allocated_milli=0, gpu_type="T4")]),
        Node(node_id="n1", cpu_total=2000, memory_total=4096, model="T4", gpu_list=[GPU(allocated_milli=0, gpu_type="T4")]),
    ]
    pods = [
        Pod(name="p0", cpu_milli=100, memory_mib=100, num_gpu=1, gpu_milli=1000, gpu_spec=("T4",), qos="Guaranteed", priority=1, creation_time=0, deletion_time=600000),
        Pod(name="p1", cpu_milli=100, memory_mib=100, num_gpu=1, gpu_milli=1000, gpu_spec=("T4",), qos="BestEffort", priority=0, creation_time=1000, deletion_time=610000),
        Pod(name="p2", cpu_milli=100, memory_mib=100, num_gpu=1, gpu_milli=1000, gpu_spec=("T4",), qos="Burstable", priority=0, creation_time=2000, deletion_time=620000),
    ]
    return nodes, pods


def run_ablation(seed: int = 42) -> Dict[str, Dict]:
    nodes, pods = make_minimal_cluster()

    ablations = {
        "success_only": {
            "frag_weight": 0.0,
            "balance_weight": 0.0,
            "util_weight": 0.0,
            "global_frag_weight": 0.0,
            "free_gpu_weight": 0.0,
            "reward_mode": "legacy",
            "slo_penalty": 0.0,
        },
        "frag_only": {
            "frag_weight": 2.0,
            "balance_weight": 0.0,
            "util_weight": 0.0,
            "global_frag_weight": 0.0,
            "free_gpu_weight": 0.0,
            "reward_mode": "legacy",
            "slo_penalty": 0.0,
        },
        "full": {
            # use latency objective by default (no shaping)
            "reward_mode": "latency",
        },
    }

    results: Dict[str, Dict] = {}
    for name, cfg in ablations.items():
        env = GPUSchedulingEnv(
            nodes=nodes,
            pods=pods,
            demand_distribution={},
            node_count=len(nodes),
            max_pods_per_episode=len(pods),
            seed=seed,
            **cfg,
        )
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            mask = env.action_masks()
            idxs = np.flatnonzero(mask)
            action = int(idxs[0]) if idxs.size else 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1

        metrics = info.get("episode_metrics", {}) if isinstance(info, dict) else {}
        results[name] = {
            "steps": steps,
            "reward_sum": float(env._episode_reward_sum),
            "metrics": metrics,
        }

    return results


def main() -> None:
    out = run_ablation(seed=123)
    p = Path("experiments/ablation_smoke_results.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"Wrote results to {p}")


if __name__ == "__main__":
    main()
