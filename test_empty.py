from train_maskable_ppo import EnvParams, make_env_factory
from pathlib import Path

env = make_env_factory(EnvParams(
    nodes_csv=Path("preprocessed/phase3/nodes_clean.csv"),
    pods_csv=Path("preprocessed/phase3/pods_test.csv"),
    demand_json=Path("preprocessed/phase3/workload_demand_distribution.json"),
    node_count=64,
    cpu_only_node_count=0,
    episode_len=1000,
    seed=42,
    env_rank=0,
    n_envs=1,
    fragmentation_mode="fgd",
    fragmentation_scale=1.0,
    frag_delta_scale=100.0,
    gpu_capacity_scale=1.0,
    frag_weight=120.0,
    balance_weight=0.75,
    util_weight=0.0,
    global_frag_weight=12.0,
    free_gpu_weight=0.03,
    free_gpu_penalty_mode="terminal",
    reward_mode="latency",
    success_reward=1.0,
    fail_penalty=-5.0,
    slo_penalty=-2.0,
    slo_threshold_ms=30000,
    priority_multiplier=3.0,
    sub_placement_policy="most_used_first",
    episode_order_mode="sequential",
    gpu_only_pods=True,
    pod_replication_factor=1,
    min_pod_duration_ms=600000,
))().env

obs, info = env.reset()
done = False
steps = 0
while not done:
    mask = env.unwrapped.strict_action_mask()
    valid = [i for i, m in enumerate(mask) if m]
    action = valid[0] if valid else 0
    obs, r, term, trunc, info = env.step(action)
    done = term or trunc
    steps += 1
print(f"Episode done in {steps} steps.")
