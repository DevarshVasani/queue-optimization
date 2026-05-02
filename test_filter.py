from gpu_scheduling_env import load_nodes
from pathlib import Path
import csv
from gpu_scheduling_env import build_env_from_preprocessed_csv

env = build_env_from_preprocessed_csv(
    nodes_csv=Path("preprocessed/phase3/nodes_clean.csv"),
    pods_csv=Path("preprocessed/phase3/pods_test.csv"),
    demand_distribution_json=Path("preprocessed/phase3/workload_demand_distribution.json"),
    node_count=64,
    gpu_only_pods=True,
    pod_replication_factor=1,
)
print(f"Number of episodes: {len(env.episodes)}")
print(f"Number of pods in episode 0: {len(env._episode_pods)}")
