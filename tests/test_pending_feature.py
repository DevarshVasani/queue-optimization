from cluster_sim import Node, GPU, Pod
from gpu_scheduling_env import GPUSchedulingEnv


def test_pods_pending_feature():
    nodes = [
        Node(node_id="n0", cpu_total=1000, memory_total=1024, model="T4", gpu_list=[GPU(allocated_milli=0, gpu_type="T4")]),
        Node(node_id="n1", cpu_total=1000, memory_total=1024, model="T4", gpu_list=[GPU(allocated_milli=0, gpu_type="T4")]),
    ]
    pods = [
        Pod(name="p0", cpu_milli=100, memory_mib=100, num_gpu=1, gpu_milli=1000, gpu_spec=("T4",), qos="Guaranteed", priority=1, creation_time=0, deletion_time=600000),
        Pod(name="p1", cpu_milli=100, memory_mib=100, num_gpu=1, gpu_milli=1000, gpu_spec=("T4",), qos="BestEffort", priority=0, creation_time=1000, deletion_time=610000),
    ]

    env = GPUSchedulingEnv(nodes=nodes, pods=pods, demand_distribution={}, node_count=2, max_pods_per_episode=2, seed=1)
    obs, info = env.reset()
    # call _global_features directly
    gf = env._global_features()
    # per _global_features: per_model_free (len GPU_MODEL_ORDER), frag_avg, pods_pending, current_time_norm
    pods_pending_value = gf[len(gf) - 2]
    expected = float(len(env._pending_pods)) / float(max(1, env._episode_limit()))
    assert abs(pods_pending_value - expected) < 1e-6


if __name__ == "__main__":
    test_pods_pending_feature()
    print("test passed")
