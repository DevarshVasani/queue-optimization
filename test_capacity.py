import pandas as pd
nodes = pd.read_csv("preprocessed/phase3/nodes_clean.csv")
pods = pd.read_csv("preprocessed/phase3/pods_test.csv")

max_node_cpu = nodes['cpu_milli'].max()
max_node_mem = nodes['memory_mib'].max()
max_node_gpu = nodes['gpu'].max() * 1000  # assuming 1000milli per GPU

too_big = pods[
    (pods['cpu_milli'] > max_node_cpu) |
    (pods['memory_mib'] > max_node_mem) |
    (pods['total_gpu_milli'] > max_node_gpu)
]
print(f"Pods too big: {len(too_big)} / {len(pods)}")
