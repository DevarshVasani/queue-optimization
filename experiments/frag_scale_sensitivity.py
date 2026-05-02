from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from cluster_sim import load_nodes, load_pods, ClusterSimulator, policy_from_name


def run_scale_experiments(*, node_csv: Path, pod_csv: Path, scales: List[float], policies: List[str], out_dir: Path, pod_limit: int = 200) -> Dict:
    nodes = load_nodes(node_csv)
    pods = load_pods(pod_csv)
    if pod_limit is not None and pod_limit > 0:
        pods = pods[:pod_limit]
    results: Dict[str, Dict[float, Dict]] = {p: {} for p in policies}

    for scale in scales:
        sim = ClusterSimulator(nodes=nodes, pods=pods, fragmentation_scale=scale, record_history=False)
        for policy_name in policies:
            policy = policy_from_name(policy_name)
            summary = sim.run_policy(policy)
            results[policy_name][scale] = {
                "cluster_fragmentation_score": float(summary.get("cluster_fragmentation_score", 0.0)),
                "avg_jct": float(summary.get("avg_jct", 0.0)),
                "gpu_allocation_ratio": float(summary.get("gpu_allocation_ratio", 0.0)),
            }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "frag_scale_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_results(results: Dict, out_dir: Path) -> None:
    scales = sorted(next(iter(results.values())).keys())
    for metric in ["cluster_fragmentation_score", "avg_jct", "gpu_allocation_ratio"]:
        plt.figure(figsize=(6, 3))
        for policy, vals in results.items():
            ys = [vals[s][metric] for s in scales]
            plt.plot(scales, ys, marker="o", label=policy)
        plt.xlabel("fragmentation_scale")
        plt.ylabel(metric)
        plt.xscale("log")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_vs_fragscale.png", dpi=150)
        plt.close()


def main() -> None:
    node_csv = Path("preprocessed/phase3/nodes_clean.csv")
    pod_csv = Path("preprocessed/phase3/pods_test.csv")
    scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    policies = ["fgd", "best_fit", "dot_product"]
    out_dir = Path("experiments/frag_scale_plots")

    results = run_scale_experiments(node_csv=node_csv, pod_csv=pod_csv, scales=scales, policies=policies, out_dir=out_dir, pod_limit=50)
    plot_results(results, out_dir=out_dir)
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
