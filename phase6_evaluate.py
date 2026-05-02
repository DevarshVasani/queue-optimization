from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cluster_sim import Node, Pod, load_nodes, load_pods
from gpu_scheduling_env import GPUSchedulingEnv


DEFAULT_BASELINES = ["random", "best_fit", "dot_product", "gpu_packing", "gpu_clustering", "fgd"]
DEFAULT_SCENARIOS = ["full", "high_load", "gpu_intensive", "mixed", "heterogeneous"]


@dataclass
class Scenario:
    name: str
    description: str
    nodes: List[Node]
    pods: List[Pod]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 evaluation for RL vs baselines on held-out traces")
    parser.add_argument("--nodes-csv", type=Path, default=Path("preprocessed/phase3/nodes_clean.csv"))
    parser.add_argument("--pods-train-csv", type=Path, default=Path("preprocessed/phase3/pods_train.csv"))
    parser.add_argument("--pods-test-csv", type=Path, default=Path("preprocessed/phase3/pods_test.csv"))
    parser.add_argument(
        "--demand-json",
        type=Path,
        default=Path("preprocessed/phase3/workload_demand_distribution.json"),
        help="Demand distribution JSON used by fragmentation scoring",
    )

    parser.add_argument("--node-count", type=int, default=128)
    parser.add_argument("--cpu-only-node-count", type=int, default=16)
    parser.add_argument(
        "--node-selection-policy",
        type=str,
        default="capacity",
        choices=["capacity", "stratified_gpu"],
        help="How to choose the evaluation cluster from nodes_csv",
    )
    parser.add_argument("--sub-placement-policy", type=str, default="most_used_first")
    parser.add_argument("--util-weight", type=float, default=0.0)
    parser.add_argument("--gpu-only-pods", action="store_true", help="Filter pods to num_gpu > 0 before scenarios")
    parser.add_argument(
        "--pod-replication-factor",
        type=int,
        default=1,
        help="Replicate the pod trace this many times to increase demand pressure",
    )
    parser.add_argument(
        "--min-pod-duration-ms",
        type=int,
        default=600_000,
        help="Enforce minimum pod lifetime to create overlap and contention",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma list from: full,high_load,gpu_intensive,mixed,heterogeneous",
    )
    parser.add_argument("--high-load-reduction-frac", type=float, default=0.3)
    parser.add_argument(
        "--high-load-mode",
        type=str,
        default="reduce_per_node",
        choices=["reduce_per_node", "drop_gpu_nodes"],
        help="reduce_per_node keeps action space fixed; drop_gpu_nodes removes GPU nodes",
    )
    parser.add_argument("--gpu-intensive-min-gpus", type=int, default=4)
    parser.add_argument(
        "--heterogeneous-models",
        type=str,
        default="A100,V100M32",
        help="Comma-separated GPU models for heterogeneous filter",
    )

    parser.add_argument("--run-rl", action="store_true", help="Run RL checkpoints in evaluation")
    parser.add_argument("--rl-device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--rl-experiments", type=str, default="full", help="Comma list, e.g. success_only,frag_only,full")
    parser.add_argument("--models-root", type=Path, default=Path("models/phase5"))

    parser.add_argument("--run-baselines", action="store_true", help="Run heuristic baselines")
    parser.add_argument("--baselines", type=str, default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--random-runs", type=int, default=5, help="Number of random baseline repeats")

    parser.add_argument("--temporal-gap", action="store_true", help="Also evaluate selected methods on train trace")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/phase6"))
    parser.add_argument("--make-plots", action="store_true")

    parser.add_argument("--frag-weight", type=float, default=2.0)
    parser.add_argument("--balance-weight", type=float, default=0.5)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--fail-penalty", type=float, default=-5.0)
    parser.add_argument("--reward-mode", type=str, default="latency", choices=["latency", "legacy"])
    parser.add_argument("--slo-penalty", type=float, default=-2.0)
    parser.add_argument("--slo-threshold-ms", type=int, default=30_000)
    parser.add_argument("--priority-multiplier", type=float, default=3.0)

    args = parser.parse_args()
    if not args.run_rl and not args.run_baselines:
        args.run_rl = True
        args.run_baselines = True
    return args


def load_demand_distribution(path: Path) -> Dict[str, Dict[Tuple[int, int], float]]:
    with path.open("r") as f:
        payload = json.load(f)
    dist_raw = payload.get("distribution", payload)
    out: Dict[str, Dict[Tuple[int, int], float]] = {}
    for model, model_dist in dist_raw.items():
        parsed: Dict[Tuple[int, int], float] = {}
        for key, prob in model_dist.items():
            if isinstance(key, str) and "|" in key:
                a, b = key.split("|", 1)
                parsed[(int(a), int(b))] = float(prob)
            elif isinstance(key, (tuple, list)) and len(key) == 2:
                parsed[(int(key[0]), int(key[1]))] = float(prob)
        out[str(model)] = parsed
    return out


def select_nodes(
    nodes: Sequence[Node],
    *,
    node_count: int,
    cpu_only_node_count: int,
    policy: str,
    seed: int,
) -> List[Node]:
    gpu_nodes = [n for n in nodes if n.gpu_count > 0]
    cpu_nodes = [n for n in nodes if n.gpu_count == 0]

    policy_name = policy.strip().lower()
    if policy_name == "stratified_gpu":
        rng = np.random.default_rng(seed)
        gpu_budget = max(0, node_count - max(0, cpu_only_node_count))
        gpu_buckets: Dict[str, List[Node]] = {}
        for node in gpu_nodes:
            gpu_buckets.setdefault(node.model, []).append(node)

        for bucket in gpu_buckets.values():
            bucket.sort(key=lambda n: (-n.gpu_count, n.node_id))

        selected_gpu: List[Node] = []
        model_order = sorted(gpu_buckets.keys(), key=lambda model: (-len(gpu_buckets[model]), model))
        if model_order:
            model_order = list(rng.permutation(model_order))
        while len(selected_gpu) < gpu_budget and model_order:
            progressed = False
            for model in list(model_order):
                bucket = gpu_buckets.get(model, [])
                if not bucket:
                    continue
                selected_gpu.append(bucket.pop(0))
                progressed = True
                if len(selected_gpu) >= gpu_budget:
                    break
            model_order = [model for model in model_order if gpu_buckets.get(model)]
            if not progressed:
                break

        if len(selected_gpu) < gpu_budget:
            leftovers = [node for bucket in gpu_buckets.values() for node in bucket]
            leftovers.sort(key=lambda n: (-n.gpu_count, n.node_id))
            selected_gpu.extend(leftovers[: gpu_budget - len(selected_gpu)])

        selected: List[Node] = list(selected_gpu)
        remaining = node_count - len(selected)
        if remaining > 0:
            cpu_nodes_sorted = sorted(cpu_nodes, key=lambda n: n.node_id)
            selected.extend(cpu_nodes_sorted[:remaining])

        if len(selected) < node_count:
            leftovers = [n for n in gpu_nodes if n not in selected]
            leftovers.sort(key=lambda n: (-n.gpu_count, n.node_id))
            selected.extend(leftovers[: node_count - len(selected)])

        if len(selected) != node_count:
            raise ValueError(f"unable to select node_count={node_count} from available={len(nodes)}")
        return sorted(copy.deepcopy(selected), key=lambda n: n.node_id)

    gpu_nodes_sorted = sorted(gpu_nodes, key=lambda n: (-n.gpu_count, n.node_id))
    cpu_nodes_sorted = sorted(cpu_nodes, key=lambda n: n.node_id)

    gpu_budget = max(0, node_count - max(0, cpu_only_node_count))
    selected: List[Node] = []
    selected.extend(gpu_nodes_sorted[:gpu_budget])

    remaining = node_count - len(selected)
    if remaining > 0:
        selected.extend(cpu_nodes_sorted[:remaining])

    if len(selected) < node_count:
        leftovers = [n for n in gpu_nodes_sorted[gpu_budget:] if n not in selected]
        selected.extend(leftovers[: node_count - len(selected)])

    if len(selected) != node_count:
        raise ValueError(f"unable to select node_count={node_count} from available={len(nodes)}")
    return sorted(copy.deepcopy(selected), key=lambda n: n.node_id)


def reduce_gpu_capacity_per_node(nodes: Sequence[Node], reduction_frac: float) -> List[Node]:
    frac = min(max(float(reduction_frac), 0.0), 0.95)
    out = copy.deepcopy(list(nodes))
    for node in out:
        if node.gpu_count <= 0:
            continue
        keep = max(1, int(math.floor(node.gpu_count * (1.0 - frac))))
        node.gpu_list = node.gpu_list[:keep]
        node.reset()
    return out


def drop_gpu_nodes(nodes: Sequence[Node], reduction_frac: float, seed: int) -> List[Node]:
    frac = min(max(float(reduction_frac), 0.0), 0.95)
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(list(nodes))
    gpu_idx = [i for i, n in enumerate(out) if n.gpu_count > 0]
    if not gpu_idx:
        return out
    drop_n = int(round(len(gpu_idx) * frac))
    if drop_n <= 0:
        return out
    drop_idx = set(int(i) for i in rng.choice(gpu_idx, size=min(drop_n, len(gpu_idx)), replace=False))
    kept = [n for i, n in enumerate(out) if i not in drop_idx]
    for n in kept:
        n.reset()
    return sorted(kept, key=lambda n: n.node_id)


def create_env(
    *,
    nodes: Sequence[Node],
    pods: Sequence[Pod],
    demand_distribution: Dict[str, Dict[Tuple[int, int], float]],
    seed: int,
    args: argparse.Namespace,
) -> GPUSchedulingEnv:
    if len(pods) <= 0:
        raise ValueError("pods list is empty")
    cpu_only_count = sum(1 for n in nodes if n.gpu_count == 0)
    return GPUSchedulingEnv(
        nodes=nodes,
        pods=pods,
        demand_distribution=demand_distribution,
        node_count=len(nodes),
        cpu_only_node_count=cpu_only_count,
        max_pods_per_episode=len(pods),
        seed=seed,
        frag_weight=args.frag_weight,
        balance_weight=args.balance_weight,
        util_weight=args.util_weight,
        reward_mode=args.reward_mode,
        success_reward=args.success_reward,
        fail_penalty=args.fail_penalty,
        slo_penalty=args.slo_penalty,
        slo_threshold_ms=args.slo_threshold_ms,
        priority_multiplier=args.priority_multiplier,
        sub_placement_policy=args.sub_placement_policy,
        min_pod_duration_ms=args.min_pod_duration_ms,
        invalid_action_mode="error",
        episode_order_mode="sequential",
    )


def _feasible_indices(mask: np.ndarray) -> List[int]:
    return [int(i) for i in np.flatnonzero(mask)]


def _simulate_node_after(
    env: GPUSchedulingEnv,
    node_idx: int,
) -> Optional[Node]:
    sim = env.sim
    if sim is None:
        return None
    pod = env._current_pod()
    node = sim.nodes[node_idx]
    candidates = sim._candidate_gpu_sets(node, pod)  # type: ignore[attr-defined]
    gpu_idxs = sim._select_gpu_indices_for_policy(  # type: ignore[attr-defined]
        node=node,
        pod=pod,
        candidates=candidates,
        policy=env.sub_placement_policy,
    )
    if len(gpu_idxs) != pod.num_gpu:
        return None
    return sim.simulate_node_after(node, pod, gpu_idxs)


def choose_baseline_action(
    *,
    policy_name: str,
    env: GPUSchedulingEnv,
    strict_mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    feasible = _feasible_indices(strict_mask)
    if not feasible:
        return 0
    if env.sim is None:
        return feasible[0]
    pod = env._current_pod()
    sim = env.sim
    key = policy_name.strip().lower()

    if key == "random":
        return int(rng.choice(feasible))

    if key == "best_fit":
        best: Optional[Tuple[Tuple[int, int, str], int]] = None
        for idx in feasible:
            node = sim.nodes[idx]
            remain = node.free_gpu_milli - pod.total_gpu_milli
            cpu_remain = node.cpu_avail - pod.cpu_milli
            score = (remain, cpu_remain, node.node_id)
            if best is None or score < best[0]:
                best = (score, idx)
        return feasible[0] if best is None else best[1]

    if key == "dot_product":
        best_score: Optional[float] = None
        best_idx = feasible[0]
        pod_vec = (pod.cpu_milli, pod.memory_mib, pod.total_gpu_milli, pod.gpu_milli)
        for idx in feasible:
            node = sim.nodes[idx]
            node_vec = (
                node.cpu_avail,
                node.memory_avail,
                node.free_gpu_milli,
                max((g.free_milli for g in node.gpu_list), default=0),
            )
            score = float(sum(a * b for a, b in zip(pod_vec, node_vec)))
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    if key == "gpu_packing":
        best: Optional[Tuple[Tuple[int, int, str], int]] = None
        for idx in feasible:
            node_after = _simulate_node_after(env, idx)
            if node_after is None:
                continue
            partial = node_after.partially_used_gpu_count
            free = node_after.free_gpu_milli
            score = (-partial, free, node_after.node_id)
            if best is None or score < best[0]:
                best = (score, idx)
        return feasible[0] if best is None else best[1]

    if key == "gpu_clustering":
        gpu_required = pod.num_gpu > 0
        preferred: List[int] = []
        fallback: List[int] = []
        for idx in feasible:
            node = sim.nodes[idx]
            has_gpu = node.gpu_count > 0
            if gpu_required and has_gpu:
                preferred.append(idx)
            elif (not gpu_required) and (not has_gpu):
                preferred.append(idx)
            else:
                fallback.append(idx)
        target = preferred if preferred else fallback
        target = sorted(target, key=lambda i: sim.nodes[i].node_id)
        return target[0] if target else feasible[0]

    if key == "fgd":
        best: Optional[Tuple[Tuple[float, int, str], int]] = None
        for idx in feasible:
            node_before = sim.nodes[idx]
            f_before = sim.get_fragmentation_score(idx)
            node_after = _simulate_node_after(env, idx)
            if node_after is None:
                continue
            f_after = sim.fragmentation_score(node_after)
            delta = f_after - f_before
            score = (delta, -node_after.cpu_avail, node_after.node_id)
            if best is None or score < best[0]:
                best = (score, idx)
        return feasible[0] if best is None else best[1]

    raise ValueError(f"Unknown baseline policy: {policy_name}")


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def _build_step_row(
    *,
    step_idx: int,
    action: int,
    reward: float,
    info: Dict[str, Any],
    capacity: float,
    used_gpu_milli: float,
) -> Dict[str, Any]:
    allocated = float(info.get("episode_allocated_gpu_milli", 0.0))
    full_free_gpu_count = float(info.get("cluster_full_free_gpu_count", 0.0))
    gar_live = 0.0 if capacity <= 0 else (used_gpu_milli / capacity)
    gar_cumulative = 0.0 if capacity <= 0 else (allocated / capacity)
    full_free_gpu_ratio = 0.0 if capacity <= 0 else (full_free_gpu_count * 1000.0 / capacity)
    return {
        "step": step_idx,
        "pod_index": int(info.get("pod_index", step_idx)),
        "pod_name": str(info.get("pod_name", "")),
        "action": int(action),
        "scheduled": 1 if bool(info.get("scheduled", False)) else 0,
        "reason": str(info.get("reason", "")),
        "reward_total": float(reward),
        "reward_success": float(info.get("reward_success", 0.0)),
        "reward_fragmentation": float(info.get("reward_fragmentation", 0.0)),
        "reward_balance": float(info.get("reward_balance", 0.0)),
        "reward_utilization": float(info.get("reward_utilization", 0.0)),
        "reward_slo": float(info.get("reward_slo", 0.0)),
        "reward_latency_objective": float(info.get("reward_latency_objective", 0.0)),
        "reward_latency_failure": float(info.get("reward_latency_failure", 0.0)),
        "frag_before": float(info.get("frag_before", 0.0)),
        "frag_after": float(info.get("frag_after", 0.0)),
        "frag_delta": float(info.get("frag_delta", 0.0)),
        "cluster_fragmentation_avg": float(info.get("cluster_fragmentation_avg", 0.0)),
        "cluster_full_free_gpu_count": full_free_gpu_count,
        "cluster_full_free_gpu_ratio": float(full_free_gpu_ratio),
        "sigma_cpu_util": float(info.get("sigma_cpu_util", 0.0)),
        "sigma_mem_util": float(info.get("sigma_mem_util", 0.0)),
        "sigma_gpu_util": float(info.get("sigma_gpu_util", 0.0)),
        "wait_time_ms": float(info.get("wait_time_ms", 0.0)),
        "job_completion_time_ms": float(info.get("job_completion_time_ms", 0.0)),
        "job_slowdown": float(info.get("job_slowdown", 0.0)),
        "latency_objective": float(info.get("latency_objective", 0.0)),
        "current_time": float(info.get("current_time", 0.0)),
        "episode_requested_gpu_milli": float(info.get("episode_requested_gpu_milli", 0.0)),
        "episode_allocated_gpu_milli": allocated,
        "used_gpu_milli": float(used_gpu_milli),
        "gar_capacity_live": float(gar_live),
        "unallocated_gpu_fraction_capacity_live": float(1.0 - gar_live),
        "gar_capacity_cumulative": float(gar_cumulative),
        "unallocated_gpu_fraction_capacity_cumulative": float(1.0 - gar_cumulative),
    }


def summarize_rows(
    *,
    method: str,
    rows: Sequence[Dict[str, Any]],
    episode_metrics: Dict[str, Any],
    infeasible_actions: int,
    capacity_milli: float,
) -> Dict[str, Any]:
    total = len(rows)
    scheduled = int(sum(int(r["scheduled"]) for r in rows))
    waits = [float(r["wait_time_ms"]) for r in rows]
    jcts = [float(r["job_completion_time_ms"]) for r in rows]
    slowdowns = [float(r["job_slowdown"]) for r in rows]
    latency_objectives = [float(r["latency_objective"]) for r in rows]
    sigma_cpu = [float(r["sigma_cpu_util"]) for r in rows]
    sigma_gpu = [float(r["sigma_gpu_util"]) for r in rows]
    frag_series = [float(r["cluster_fragmentation_avg"]) for r in rows]
    full_free_gpu_series = [float(r["cluster_full_free_gpu_count"]) for r in rows]
    gar_live_series = [float(r["gar_capacity_live"]) for r in rows]
    gar_cum_series = [float(r["gar_capacity_cumulative"]) for r in rows]

    allocated = float(episode_metrics.get("allocated_gpu_milli", rows[-1]["episode_allocated_gpu_milli"] if rows else 0.0))
    if "allocated_gpu_milli" not in episode_metrics:
        allocated = float(rows[-1]["episode_allocated_gpu_milli"]) if rows else 0.0
    gar_cumulative_final = 0.0 if capacity_milli <= 0 else allocated / capacity_milli
    gar_live_avg = float(mean(gar_live_series)) if gar_live_series else 0.0
    gar_live_final = float(gar_live_series[-1]) if gar_live_series else 0.0

    out: Dict[str, Any] = {
        "method": method,
        "total_pods": total,
        "scheduled_pods": scheduled,
        "failed_pods": total - scheduled,
        "success_rate": 0.0 if total <= 0 else (scheduled / float(total)),
        "gar_capacity": gar_live_avg,
        "gar_capacity_final": gar_live_final,
        "unallocated_gpu_fraction_capacity": 1.0 - gar_live_avg,
        "unallocated_gpu_fraction_capacity_final": 1.0 - gar_live_final,
        "gar_capacity_cumulative_final": gar_cumulative_final,
        "gar_capacity_cumulative_avg": float(mean(gar_cum_series)) if gar_cum_series else 0.0,
        "gar_requested": float(episode_metrics.get("gar_requested", 0.0)),
        "allocation_ratio_capacity_metric": float(episode_metrics.get("allocation_ratio_capacity", 0.0)),
        "avg_wait_time_ms": float(mean(waits)) if waits else 0.0,
        "p95_wait_time_ms": _percentile(waits, 95),
        "p99_wait_time_ms": _percentile(waits, 99),
        "avg_job_completion_time_ms": float(mean(jcts)) if jcts else 0.0,
        "p95_job_completion_time_ms": _percentile(jcts, 95),
        "p99_job_completion_time_ms": _percentile(jcts, 99),
        "avg_job_slowdown": float(mean(slowdowns)) if slowdowns else 0.0,
        "p95_job_slowdown": _percentile(slowdowns, 95),
        "p99_job_slowdown": _percentile(slowdowns, 99),
        "latency_objective": float(episode_metrics.get("latency_objective", latency_objectives[-1] if latency_objectives else 0.0)),
        "avg_sigma_cpu_util": float(mean(sigma_cpu)) if sigma_cpu else 0.0,
        "avg_sigma_gpu_util": float(mean(sigma_gpu)) if sigma_gpu else 0.0,
        "avg_fragmentation": float(mean(frag_series)) if frag_series else 0.0,
        "final_fragmentation": float(episode_metrics.get("final_fragmentation_avg", frag_series[-1] if frag_series else 0.0)),
        "avg_full_free_gpu_count": float(mean(full_free_gpu_series)) if full_free_gpu_series else 0.0,
        "final_full_free_gpu_count": float(full_free_gpu_series[-1]) if full_free_gpu_series else 0.0,
        "mean_frag_delta": float(episode_metrics.get("mean_frag_delta", 0.0)),
        "infeasible_action_count": int(infeasible_actions),
        "reward_sum": float(sum(float(r["reward_total"]) for r in rows)),
    }
    return out


def run_baseline(
    *,
    scenario: Scenario,
    policy_name: str,
    run_seed: int,
    demand_distribution: Dict[str, Dict[Tuple[int, int], float]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env = create_env(nodes=scenario.nodes, pods=scenario.pods, demand_distribution=demand_distribution, seed=run_seed, args=args)
    rng = np.random.default_rng(run_seed)
    obs, _ = env.reset(seed=run_seed)

    rows: List[Dict[str, Any]] = []
    capacity_milli = float(env.get_total_gpu_capacity_milli())
    infeasible_actions = 0
    done = False
    step_idx = 0
    final_info: Dict[str, Any] = {}

    while not done:
        strict_mask = env.strict_action_mask()
        action = choose_baseline_action(policy_name=policy_name, env=env, strict_mask=strict_mask, rng=rng)
        if np.any(strict_mask) and not bool(strict_mask[action]):
            infeasible_actions += 1

        obs, reward, terminated, truncated, info = env.step(action)
        row = _build_step_row(
            step_idx=step_idx,
            action=action,
            reward=float(reward),
            info=info,
            capacity=capacity_milli,
            used_gpu_milli=float(sum(n.used_gpu_milli for n in env.sim.nodes)) if env.sim is not None else 0.0,
        )
        rows.append(row)
        final_info = info
        done = bool(terminated or truncated)
        step_idx += 1

    metrics = final_info.get("episode_metrics", {}) if isinstance(final_info, dict) else {}
    summary = summarize_rows(
        method=policy_name,
        rows=rows,
        episode_metrics=metrics if isinstance(metrics, dict) else {},
        infeasible_actions=infeasible_actions,
        capacity_milli=capacity_milli,
    )
    return summary, rows


def _resolve_device(device: str) -> str:
    import torch

    d = device.strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d not in {"cpu", "cuda"}:
        raise ValueError("rl-device must be one of: auto,cpu,cuda")
    if d == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return d


def _resolve_rl_artifacts(models_root: Path, experiment: str) -> Tuple[Path, Path]:
    exp_dir = models_root / experiment
    model_path = exp_dir / "best_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"missing RL checkpoint: {model_path}")

    candidates = [
        exp_dir / "best_model.pkl",
        exp_dir / "best_vecnormalize.pkl",
        exp_dir / "vecnormalize.pkl",
    ]
    vec_path = next((p for p in candidates if p.exists()), None)
    if vec_path is None:
        raise FileNotFoundError(f"missing VecNormalize stats in {exp_dir}")
    return model_path, vec_path


def _load_vecnormalize_obs_dim(vec_path: Path) -> Optional[int]:
    try:
        with vec_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None

    obs_rms = getattr(payload, "obs_rms", None)
    mean_arr = getattr(obs_rms, "mean", None)
    if mean_arr is None:
        return None
    shape = getattr(mean_arr, "shape", None)
    if not shape:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError, IndexError):
        return None


def _maybe_expected_node_count(base_env: GPUSchedulingEnv, expected_obs_dim: int) -> Optional[int]:
    base_dim = int(base_env.pod_feature_dim + base_env.global_feature_dim)
    per_node = int(base_env.node_feature_dim)
    if per_node <= 0:
        return None
    remainder = expected_obs_dim - base_dim
    if remainder < 0 or (remainder % per_node) != 0:
        return None
    return int(remainder // per_node)


def run_rl(
    *,
    scenario: Scenario,
    experiment: str,
    run_seed: int,
    demand_distribution: Dict[str, Dict[Tuple[int, int], float]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise ImportError("Missing RL deps. Install requirements-rl.txt to run RL evaluation.") from exc

    model_path, vec_path = _resolve_rl_artifacts(args.models_root, experiment)
    base_env = create_env(nodes=scenario.nodes, pods=scenario.pods, demand_distribution=demand_distribution, seed=run_seed, args=args)

    expected_obs_dim = _load_vecnormalize_obs_dim(vec_path)
    current_obs_dim = int(base_env.observation_space.shape[0])
    if expected_obs_dim is not None and expected_obs_dim != current_obs_dim:
        expected_nodes = _maybe_expected_node_count(base_env, expected_obs_dim)
        hint = ""
        if expected_nodes is not None:
            hint = f" Try --node-count {expected_nodes} for this checkpoint or retrain with node_count={args.node_count}."
        raise ValueError(
            "RL checkpoint observation mismatch: "
            f"checkpoint obs_dim={expected_obs_dim}, eval obs_dim={current_obs_dim}, "
            f"eval node_count={len(scenario.nodes)}." + hint
        )

    vec_env = DummyVecEnv([lambda: base_env])
    vec_env = VecNormalize.load(str(vec_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    device = _resolve_device(args.rl_device)
    model = MaskablePPO.load(str(model_path), env=vec_env, device=device)

    obs = vec_env.reset()
    rows: List[Dict[str, Any]] = []
    infeasible_actions = 0
    done = False
    step_idx = 0
    final_info: Dict[str, Any] = {}
    capacity_milli = float(base_env.get_total_gpu_capacity_milli())

    while not done:
        strict_mask = vec_env.env_method("strict_action_mask")[0]
        mask = vec_env.env_method("action_masks")[0]
        mask = np.asarray(mask)
        if mask.ndim == 1:
            mask = mask.reshape(1, -1)

        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        action_idx = int(action[0]) if isinstance(action, np.ndarray) else int(action)
        if np.any(strict_mask) and not bool(strict_mask[action_idx]):
            infeasible_actions += 1

        obs, rewards, dones, infos = vec_env.step(action)
        info = infos[0] if infos else {}
        reward = float(rewards[0]) if isinstance(rewards, np.ndarray) else float(rewards)
        row = _build_step_row(
            step_idx=step_idx,
            action=action_idx,
            reward=reward,
            info=info,
            capacity=capacity_milli,
            used_gpu_milli=float(sum(n.used_gpu_milli for n in base_env.sim.nodes)) if base_env.sim is not None else 0.0,
        )
        rows.append(row)
        final_info = info
        done = bool(dones[0]) if isinstance(dones, np.ndarray) else bool(dones)
        step_idx += 1

    metrics = final_info.get("episode_metrics", {}) if isinstance(final_info, dict) else {}
    summary = summarize_rows(
        method=f"rl_{experiment}",
        rows=rows,
        episode_metrics=metrics if isinstance(metrics, dict) else {},
        infeasible_actions=infeasible_actions,
        capacity_milli=capacity_milli,
    )
    summary["model_path"] = str(model_path)
    summary["vecnormalize_path"] = str(vec_path)
    summary["device"] = device
    vec_env.close()
    return summary, rows


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(method: str, runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return {"method": method, "runs": [], "mean": {}, "std": {}}
    numeric_keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    mean_map: Dict[str, float] = {}
    std_map: Dict[str, float] = {}
    for key in numeric_keys:
        vals = [float(r[key]) for r in runs]
        mean_map[key] = float(mean(vals))
        std_map[key] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return {
        "method": method,
        "runs": list(runs),
        "mean": mean_map,
        "std": std_map,
    }


def build_scenarios(
    *,
    args: argparse.Namespace,
    base_nodes: Sequence[Node],
    test_pods: Sequence[Pod],
) -> List[Scenario]:
    wanted = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    valid = {"full", "high_load", "gpu_intensive", "mixed", "heterogeneous"}
    invalid = sorted(x for x in wanted if x not in valid)
    if invalid:
        raise ValueError(f"unknown scenarios: {invalid}")

    scenarios: List[Scenario] = []
    het_models = {x.strip() for x in args.heterogeneous_models.split(",") if x.strip()}

    for name in wanted:
        if name == "full":
            scenarios.append(
                Scenario(
                    name="full",
                    description="Unmodified held-out test trace (continuous single run)",
                    nodes=copy.deepcopy(list(base_nodes)),
                    pods=copy.deepcopy(list(test_pods)),
                )
            )
        elif name == "mixed":
            scenarios.append(
                Scenario(
                    name="mixed",
                    description="Natural mixed CPU/GPU workload (same as full test trace)",
                    nodes=copy.deepcopy(list(base_nodes)),
                    pods=copy.deepcopy(list(test_pods)),
                )
            )
        elif name == "gpu_intensive":
            pods = [p for p in test_pods if int(p.num_gpu) >= int(args.gpu_intensive_min_gpus)]
            scenarios.append(
                Scenario(
                    name="gpu_intensive",
                    description=f"GPU-intensive subset with num_gpu >= {args.gpu_intensive_min_gpus}",
                    nodes=copy.deepcopy(list(base_nodes)),
                    pods=copy.deepcopy(pods),
                )
            )
        elif name == "heterogeneous":
            pods = [p for p in test_pods if p.num_gpu > 0 and bool(set(p.gpu_spec).intersection(het_models))]
            scenarios.append(
                Scenario(
                    name="heterogeneous",
                    description=f"Pods constrained to GPU models in {sorted(het_models)}",
                    nodes=copy.deepcopy(list(base_nodes)),
                    pods=copy.deepcopy(pods),
                )
            )
        elif name == "high_load":
            if args.high_load_mode == "reduce_per_node":
                nodes = reduce_gpu_capacity_per_node(base_nodes, args.high_load_reduction_frac)
            else:
                nodes = drop_gpu_nodes(base_nodes, args.high_load_reduction_frac, args.seed + 17)
            scenarios.append(
                Scenario(
                    name="high_load",
                    description=f"High-load stress with {args.high_load_mode} reduction_frac={args.high_load_reduction_frac}",
                    nodes=nodes,
                    pods=copy.deepcopy(list(test_pods)),
                )
            )

    return scenarios


def filter_and_replicate_pods(
    pods: Sequence[Pod],
    *,
    gpu_only: bool,
    replication_factor: int,
) -> List[Pod]:
    selected = [copy.deepcopy(p) for p in pods if (not gpu_only) or (p.num_gpu > 0)]
    if not selected:
        return []

    factor = max(1, int(replication_factor))
    if factor == 1:
        return selected

    min_creation = min(p.creation_time for p in selected)
    max_deletion = max(p.deletion_time for p in selected)
    span = max(1, max_deletion - min_creation + 1)

    out: List[Pod] = []
    for rep in range(factor):
        shift = rep * span
        for pod in selected:
            p = copy.deepcopy(pod)
            p.name = f"{pod.name}__rep{rep}"
            p.creation_time = int(pod.creation_time) + shift
            p.deletion_time = int(pod.deletion_time) + shift
            p.scheduled_time = None
            out.append(p)
    return out


def _read_series(csv_path: Path, key: str) -> List[float]:
    out: List[float] = []
    if not csv_path.exists():
        return out
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(key)
            if val is None or val == "":
                continue
            out.append(float(val))
    return out


def maybe_make_plots(
    *,
    output_dir: Path,
    summary: Dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot generation")
        return

    scenarios = summary.get("scenarios", {})
    if not isinstance(scenarios, dict) or not scenarios:
        return

    # Prefer the canonical "full" scenario when present; otherwise use the
    # first available scenario so plots are still generated for focused runs.
    scenario_name = "full" if "full" in scenarios else next(iter(scenarios.keys()))
    scenario_payload = scenarios.get(scenario_name)
    if not isinstance(scenario_payload, dict):
        return

    methods = scenario_payload.get("methods", {})
    if not isinstance(methods, dict):
        return

    labels: List[str] = []
    values: List[float] = []
    errors: List[float] = []
    for method, payload in methods.items():
        mean_map = payload.get("mean", {})
        std_map = payload.get("std", {})
        if not isinstance(mean_map, dict):
            continue
        labels.append(method)
        values.append(float(mean_map.get("gar_capacity", 0.0)))
        errors.append(float(std_map.get("gar_capacity", 0.0)) if isinstance(std_map, dict) else 0.0)

    if labels:
        plt.figure(figsize=(10, 4))
        x = np.arange(len(labels))
        plt.bar(x, values, yerr=errors, capsize=4)
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("GAR (capacity-based)")
        plt.title(f"Phase 6: GAR Comparison on {scenario_name} Scenario")
        plt.tight_layout()
        path = output_dir / "gar_comparison_full.png"
        plt.savefig(path, dpi=180)
        plt.close()

    line_methods = ["rl_full", "fgd", "best_fit"]
    line_keys = [
        ("cluster_fragmentation_avg", "Fragmentation Over Time", "fragmentation_over_time_full.png"),
        ("cluster_full_free_gpu_count", "Full Free GPUs Remaining Over Time", "full_free_gpus_over_time_full.png"),
        ("sigma_gpu_util", "GPU Utilization Spread Over Time", "sigma_gpu_over_time_full.png"),
    ]

    method_to_csv: Dict[str, Path] = {}
    for m, payload in methods.items():
        runs = payload.get("runs", [])
        if runs and isinstance(runs, list) and isinstance(runs[0], dict):
            csv_path = runs[0].get("steps_csv")
            if isinstance(csv_path, str):
                method_to_csv[m] = Path(csv_path)

    for key, title, filename in line_keys:
        plt.figure(figsize=(10, 4))
        plotted = False
        for method in line_methods:
            csv_path = method_to_csv.get(method)
            if csv_path is None:
                continue
            ys = _read_series(csv_path, key)
            if not ys:
                continue
            xs = np.arange(len(ys))
            plt.plot(xs, ys, label=method)
            plotted = True
        if plotted:
            plt.title(title)
            plt.xlabel("Pods Processed")
            plt.ylabel(key)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    rl_methods = [m for m in methods.keys() if m.startswith("rl_")]
    if len(rl_methods) >= 2:
        labels = []
        vals = []
        for m in rl_methods:
            mean_map = methods[m].get("mean", {})
            labels.append(m)
            vals.append(float(mean_map.get("gar_capacity", 0.0)) if isinstance(mean_map, dict) else 0.0)
        plt.figure(figsize=(8, 4))
        x = np.arange(len(labels))
        plt.bar(x, vals)
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("GAR (capacity-based)")
        plt.title(f"RL Ablation Comparison ({scenario_name} Scenario)")
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_rl_gar_full.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    demand_distribution = load_demand_distribution(args.demand_json)
    all_nodes = load_nodes(args.nodes_csv)
    train_pods = load_pods(args.pods_train_csv, min_pod_duration=args.min_pod_duration_ms)
    test_pods = load_pods(args.pods_test_csv, min_pod_duration=args.min_pod_duration_ms)

    train_pods = filter_and_replicate_pods(
        train_pods,
        gpu_only=bool(args.gpu_only_pods),
        replication_factor=int(args.pod_replication_factor),
    )
    test_pods = filter_and_replicate_pods(
        test_pods,
        gpu_only=bool(args.gpu_only_pods),
        replication_factor=int(args.pod_replication_factor),
    )

    base_nodes = select_nodes(
        all_nodes,
        node_count=args.node_count,
        cpu_only_node_count=args.cpu_only_node_count,
        policy=args.node_selection_policy,
        seed=args.seed,
    )
    scenarios = build_scenarios(args=args, base_nodes=base_nodes, test_pods=test_pods)

    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]
    rl_experiments = [x.strip() for x in args.rl_experiments.split(",") if x.strip()]

    summary: Dict[str, Any] = {
        "config": {
            "nodes_csv": str(args.nodes_csv),
            "pods_train_csv": str(args.pods_train_csv),
            "pods_test_csv": str(args.pods_test_csv),
            "demand_json": str(args.demand_json),
            "node_count": int(args.node_count),
            "cpu_only_node_count": int(args.cpu_only_node_count),
            "node_selection_policy": args.node_selection_policy,
            "seed": int(args.seed),
            "sub_placement_policy": args.sub_placement_policy,
            "util_weight": float(args.util_weight),
            "reward_mode": args.reward_mode,
            "gpu_only_pods": bool(args.gpu_only_pods),
            "pod_replication_factor": int(args.pod_replication_factor),
            "scenarios": [s.name for s in scenarios],
            "run_rl": bool(args.run_rl),
            "run_baselines": bool(args.run_baselines),
            "rl_experiments": rl_experiments,
            "baselines": baselines,
            "random_runs": int(args.random_runs),
        },
        "scenarios": {},
    }

    table_rows: List[Dict[str, Any]] = []

    for scenario in scenarios:
        print(f"\n=== Scenario: {scenario.name} ({len(scenario.nodes)} nodes, {len(scenario.pods)} pods) ===")
        if len(scenario.pods) <= 0:
            print("Skipping: no pods after scenario filter")
            continue

        scenario_dir = output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        methods: Dict[str, Any] = {}

        if args.run_rl:
            for exp in rl_experiments:
                method_name = f"rl_{exp}"
                try:
                    run_summary, rows = run_rl(
                        scenario=scenario,
                        experiment=exp,
                        run_seed=args.seed,
                        demand_distribution=demand_distribution,
                        args=args,
                    )
                except Exception as exc:
                    print(f"[WARN] RL {method_name} skipped: {exc}")
                    continue
                csv_path = scenario_dir / f"{method_name}_run0_steps.csv"
                write_rows_csv(csv_path, rows)
                run_summary["steps_csv"] = str(csv_path)
                agg = aggregate_runs(method_name, [run_summary])
                methods[method_name] = agg

        if args.run_baselines:
            for policy in baselines:
                run_count = args.random_runs if policy == "random" else 1
                runs: List[Dict[str, Any]] = []
                for i in range(run_count):
                    run_seed = args.seed + i
                    run_summary, rows = run_baseline(
                        scenario=scenario,
                        policy_name=policy,
                        run_seed=run_seed,
                        demand_distribution=demand_distribution,
                        args=args,
                    )
                    csv_path = scenario_dir / f"{policy}_run{i}_steps.csv"
                    write_rows_csv(csv_path, rows)
                    run_summary["seed"] = int(run_seed)
                    run_summary["steps_csv"] = str(csv_path)
                    runs.append(run_summary)
                methods[policy] = aggregate_runs(policy, runs)

        comparisons: Dict[str, Any] = {}
        if "fgd" in methods and "best_fit" in methods:
            fgd_unalloc = float(methods["fgd"]["mean"].get("unallocated_gpu_fraction_capacity", 0.0))
            bf_unalloc = float(methods["best_fit"]["mean"].get("unallocated_gpu_fraction_capacity", 0.0))
            if bf_unalloc > 0:
                comparisons["fgd_relative_unallocated_reduction_vs_best_fit"] = (bf_unalloc - fgd_unalloc) / bf_unalloc
            else:
                comparisons["fgd_relative_unallocated_reduction_vs_best_fit"] = 0.0

        if "rl_full" in methods and "fgd" in methods and scenario.name == "full":
            rl_gar = float(methods["rl_full"]["mean"].get("gar_capacity", 0.0))
            fgd_gar = float(methods["fgd"]["mean"].get("gar_capacity", 0.0))
            comparisons["rl_full_vs_fgd_gar_delta_abs"] = rl_gar - fgd_gar
            comparisons["rl_full_vs_fgd_gar_delta_pct"] = 0.0 if fgd_gar <= 0 else ((rl_gar - fgd_gar) / fgd_gar)

        summary["scenarios"][scenario.name] = {
            "description": scenario.description,
            "num_nodes": len(scenario.nodes),
            "num_pods": len(scenario.pods),
            "methods": methods,
            "comparisons": comparisons,
        }

        for method, payload in methods.items():
            row = {
                "scenario": scenario.name,
                "method": method,
                "gar_capacity_mean": float(payload["mean"].get("gar_capacity", 0.0)),
                "gar_capacity_std": float(payload["std"].get("gar_capacity", 0.0)),
                "unallocated_mean": float(payload["mean"].get("unallocated_gpu_fraction_capacity", 0.0)),
                "success_rate_mean": float(payload["mean"].get("success_rate", 0.0)),
                "avg_sigma_gpu_util_mean": float(payload["mean"].get("avg_sigma_gpu_util", 0.0)),
                "avg_wait_time_ms_mean": float(payload["mean"].get("avg_wait_time_ms", 0.0)),
                "avg_job_completion_time_ms_mean": float(payload["mean"].get("avg_job_completion_time_ms", 0.0)),
                "p95_job_completion_time_ms_mean": float(payload["mean"].get("p95_job_completion_time_ms", 0.0)),
                "p99_job_completion_time_ms_mean": float(payload["mean"].get("p99_job_completion_time_ms", 0.0)),
                "latency_objective_mean": float(payload["mean"].get("latency_objective", 0.0)),
                "final_full_free_gpu_count_mean": float(payload["mean"].get("final_full_free_gpu_count", 0.0)),
                "infeasible_action_count_mean": float(payload["mean"].get("infeasible_action_count", 0.0)),
            }
            table_rows.append(row)

    if args.temporal_gap:
        print("\n=== Temporal Gap Check (train vs test) ===")
        temporal: Dict[str, Any] = {}
        train_scenario = Scenario(
            name="train_full",
            description="Training trace full run",
            nodes=copy.deepcopy(base_nodes),
            pods=copy.deepcopy(train_pods),
        )
        test_full = summary.get("scenarios", {}).get("full")
        if isinstance(test_full, dict):
            methods_to_check: List[str] = []
            if args.run_rl and "rl_full" in test_full.get("methods", {}):
                methods_to_check.append("rl_full")
            if args.run_baselines and "fgd" in test_full.get("methods", {}):
                methods_to_check.append("fgd")

            for method in methods_to_check:
                if method.startswith("rl_"):
                    exp = method.replace("rl_", "", 1)
                    run_summary, _ = run_rl(
                        scenario=train_scenario,
                        experiment=exp,
                        run_seed=args.seed,
                        demand_distribution=demand_distribution,
                        args=args,
                    )
                else:
                    run_summary, _ = run_baseline(
                        scenario=train_scenario,
                        policy_name=method,
                        run_seed=args.seed,
                        demand_distribution=demand_distribution,
                        args=args,
                    )
                test_gar = float(test_full["methods"][method]["mean"].get("gar_capacity", 0.0))
                train_gar = float(run_summary.get("gar_capacity", 0.0))
                temporal[method] = {
                    "train_gar_capacity": train_gar,
                    "test_gar_capacity": test_gar,
                    "gap_abs": test_gar - train_gar,
                    "gap_pct_vs_train": 0.0 if train_gar <= 0 else ((test_gar - train_gar) / train_gar),
                }
        summary["temporal_gap"] = temporal

    summary_path = output_dir / "phase6_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")

    if table_rows:
        table_path = output_dir / "phase6_results_table.csv"
        with table_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            writer.writeheader()
            writer.writerows(table_rows)
        print(f"Wrote table: {table_path}")

    if args.make_plots:
        maybe_make_plots(output_dir=output_dir, summary=summary)
        print(f"Plots saved under: {output_dir}")


if __name__ == "__main__":
    main()
