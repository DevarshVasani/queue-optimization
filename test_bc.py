import argparse
from pathlib import Path
from train_maskable_ppo import _build_bc_dataset

args = argparse.Namespace(
    nodes_csv=Path("preprocessed/phase3/nodes_clean.csv"),
    train_pods_csv=Path("preprocessed/phase3/pods_test.csv"),
    demand_json=Path("preprocessed/phase3/workload_demand_distribution.json"),
    node_count=64,
    cpu_only_node_count=0,
    episode_len=5000,
    seed=42,
    fragmentation_mode="fgd",
    fragmentation_scale=1.0,
    frag_delta_scale=100.0,
    gpu_capacity_scale=1.0,
    reward_mode="latency",
    slo_threshold_ms=30000,
    priority_multiplier=3.0,
    sub_placement_policy="most_used_first",
    gpu_only_pods=True,
    min_pod_duration_ms=600000,
    bc_trace_fraction=0.5,
    bc_max_samples=20000,
)
reward_cfg = {
    "frag_weight": 120.0,
    "balance_weight": 0.75,
    "util_weight": 0.0,
    "global_frag_weight": 12.0,
    "free_gpu_weight": 0.03,
    "free_gpu_penalty_mode": "terminal",
    "success_reward": 1.0,
    "fail_penalty": -5.0,
    "slo_penalty": -2.0,
    "reward_mode": "latency"
}

obs, acts, masks = _build_bc_dataset(args=args, reward_cfg=reward_cfg, replication_factor=1)
print(f"Dataset shape: {obs.shape}")
