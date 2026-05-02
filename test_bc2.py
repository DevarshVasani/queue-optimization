import argparse
from pathlib import Path
from train_maskable_ppo import _build_bc_dataset, parse_args

args = parse_args()
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
obs, acts, masks = _build_bc_dataset(args=args, reward_cfg=reward_cfg, replication_factor=5)
print(f"Dataset shape: {obs.shape}")
