from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing RL dependencies. Install with: pip install -r requirements-rl.txt"
    ) from exc

from gpu_scheduling_env import GPUSchedulingEnv, build_env_from_preprocessed_csv


def mask_fn(env: GPUSchedulingEnv) -> np.ndarray:
    return env.action_masks()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    d = device.strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if d == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return d


@dataclass
class EnvParams:
    nodes_csv: Path
    pods_csv: Path
    demand_json: Path
    node_count: int
    cpu_only_node_count: int
    episode_len: int
    seed: int
    env_rank: int
    n_envs: int
    fragmentation_mode: str
    fragmentation_scale: float
    frag_delta_scale: float
    gpu_capacity_scale: float
    frag_weight: float
    balance_weight: float
    util_weight: float
    global_frag_weight: float
    free_gpu_weight: float
    free_gpu_penalty_mode: str
    success_reward: float
    fail_penalty: float
    slo_penalty: float
    slo_threshold_ms: int
    priority_multiplier: float
    sub_placement_policy: str
    episode_order_mode: str
    gpu_only_pods: bool
    pod_replication_factor: int
    min_pod_duration_ms: int


class NodeAttentionExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        node_count: int,
        node_feature_dim: int,
        pod_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_attn_layers: int = 2,
    ) -> None:
        super().__init__(observation_space, features_dim=hidden_dim)
        self.node_count = int(node_count)
        self.node_feature_dim = int(node_feature_dim)
        self.pod_feature_dim = int(pod_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)

        node_input_dim = self.node_feature_dim + self.pod_feature_dim
        self.node_embed = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(max(1, num_attn_layers))]
        )
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(1, num_attn_layers))])
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim + self.pod_feature_dim + self.global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pod_end = self.pod_feature_dim
        node_end = pod_end + (self.node_count * self.node_feature_dim)

        pod_features = obs[:, :pod_end]
        node_flat = obs[:, pod_end:node_end]
        global_features = obs[:, node_end : node_end + self.global_feature_dim]

        node_features = node_flat.view(-1, self.node_count, self.node_feature_dim)
        pod_tiled = pod_features.unsqueeze(1).expand(-1, self.node_count, -1)
        node_input = torch.cat([node_features, pod_tiled], dim=-1)

        hidden = self.node_embed(node_input)
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            attn_out, _ = attn(hidden, hidden, hidden, need_weights=False)
            hidden = norm(hidden + attn_out)

        pooled = hidden.mean(dim=1)
        combined = torch.cat([pooled, pod_features, global_features], dim=-1)
        return self.combined(combined)


class ValidationGARCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env: VecNormalize,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_dir: Path,
        patience_steps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(max(1, eval_freq))
        self.n_eval_episodes = int(max(1, n_eval_episodes))
        self.best_model_dir = best_model_dir
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.patience_steps = int(max(0, patience_steps))
        self.best_gar = float("-inf")
        self.best_step = 0

    def _evaluate(self) -> Dict[str, float]:
        gar_vals: List[float] = []
        allocation_ratio_capacity_vals: List[float] = []
        unallocated_vals: List[float] = []
        success_vals: List[float] = []
        frag_delta_vals: List[float] = []
        frag_final_vals: List[float] = []
        sigma_cpu_vals: List[float] = []
        sigma_mem_vals: List[float] = []
        sigma_gpu_vals: List[float] = []
        wait_time_vals: List[float] = []
        reward_vals: List[float] = []
        free_gpu_vals: List[float] = []

        self.eval_env.training = False
        obs = self.eval_env.reset()
        episodes_done = 0

        while episodes_done < self.n_eval_episodes:
            mask = self.eval_env.env_method("action_masks")[0]
            mask = np.asarray(mask)
            if mask.ndim == 1:
                mask = mask.reshape(1, -1)
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, dones, infos = self.eval_env.step(action)
            done = bool(dones[0])
            if not done:
                continue
            info = infos[0]
            metrics = info.get("episode_metrics", {}) if isinstance(info, dict) else {}
            if isinstance(metrics, dict):
                gar_vals.append(float(metrics.get("gar", 0.0)))
                allocation_ratio_capacity_vals.append(float(metrics.get("allocation_ratio_capacity", 0.0)))
                
                unallocated_vals.append(float(metrics.get("unallocated_gpu_fraction", 0.0)))
                success_vals.append(float(metrics.get("success_rate", 0.0)))
                frag_delta_vals.append(float(metrics.get("mean_frag_delta", 0.0)))
                frag_final_vals.append(float(metrics.get("final_fragmentation_avg", 0.0)))
                sigma_cpu_vals.append(float(metrics.get("sigma_cpu_util", 0.0)))
                sigma_mem_vals.append(float(metrics.get("sigma_mem_util", 0.0)))
                sigma_gpu_vals.append(float(metrics.get("sigma_gpu_util", 0.0)))
                wait_time_vals.append(float(metrics.get("mean_wait_time_ms", 0.0)))
                reward_vals.append(float(metrics.get("episode_reward", 0.0)))
                free_gpu_vals.append(float(metrics.get("full_free_gpu_count", 0.0)))
            episodes_done += 1

        def _mean(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        return {
            "gar": _mean(gar_vals),
            "allocation_ratio_capacity": _mean(allocation_ratio_capacity_vals),
            "unallocated_gpu_fraction": _mean(unallocated_vals),
            "success_rate": _mean(success_vals),
            "mean_frag_delta": _mean(frag_delta_vals),
            "final_fragmentation_avg": _mean(frag_final_vals),
            "sigma_cpu_util": _mean(sigma_cpu_vals),
            "sigma_mem_util": _mean(sigma_mem_vals),
            "sigma_gpu_util": _mean(sigma_gpu_vals),
            "mean_wait_time_ms": _mean(wait_time_vals),
            "episode_reward": _mean(reward_vals),
            "full_free_gpu_count": _mean(free_gpu_vals),
        }

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        if isinstance(self.training_env, VecNormalize):
            self.eval_env.obs_rms = self.training_env.obs_rms

        metrics = self._evaluate()
        self.logger.record("validation/gar", metrics["gar"])
        self.logger.record("validation/gar_capacity", metrics["allocation_ratio_capacity"])
        self.logger.record("validation/allocation_ratio_capacity", metrics["allocation_ratio_capacity"])
        self.logger.record("validation/unallocated_gpu_fraction", metrics["unallocated_gpu_fraction"])
        self.logger.record("validation/success_rate", metrics["success_rate"])
        self.logger.record("validation/mean_frag_delta", metrics["mean_frag_delta"])
        self.logger.record("validation/final_fragmentation_avg", metrics["final_fragmentation_avg"])
        self.logger.record("validation/sigma_cpu_util", metrics["sigma_cpu_util"])
        self.logger.record("validation/sigma_mem_util", metrics["sigma_mem_util"])
        self.logger.record("validation/sigma_gpu_util", metrics["sigma_gpu_util"])
        self.logger.record("validation/full_free_gpu_count", metrics["full_free_gpu_count"])
        self.logger.record("validation/mean_wait_time_ms", metrics["mean_wait_time_ms"])
        self.logger.record("validation/episode_reward", metrics["episode_reward"])

        current_step = int(self.num_timesteps)
        improved = metrics["allocation_ratio_capacity"] > self.best_gar
        if improved:
            self.best_gar = metrics["allocation_ratio_capacity"]
            self.best_step = current_step
            self.model.save(str(self.best_model_dir / "best_model"))
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(str(self.best_model_dir / "best_vecnormalize.pkl"))
                self.training_env.save(str(self.best_model_dir / "best_model.pkl"))

            payload = {
                "best_gar": self.best_gar,
                "best_step": self.best_step,
                "metrics": metrics,
            }
            with (self.best_model_dir / "best_metrics.json").open("w") as f:
                json.dump(payload, f, indent=2)

        if self.patience_steps > 0 and (current_step - self.best_step) >= self.patience_steps:
            if self.verbose > 0:
                print(
                    f"Early stopping at step {current_step}: no allocation_ratio_capacity improvement in {self.patience_steps} steps"
                )
            return False
        return True


class SchedulingMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.episode_counter = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            self.logger.record("custom/reward_success", float(info.get("reward_success", 0.0)))
            self.logger.record("custom/reward_fragmentation", float(info.get("reward_fragmentation", 0.0)))
            self.logger.record("custom/reward_global_fragmentation", float(info.get("reward_global_fragmentation", 0.0)))
            self.logger.record("custom/reward_balance", float(info.get("reward_balance", 0.0)))
            self.logger.record("custom/reward_free_gpu", float(info.get("reward_free_gpu", 0.0)))
            self.logger.record("custom/reward_utilization", float(info.get("reward_utilization", 0.0)))
            self.logger.record("custom/reward_slo", float(info.get("reward_slo", 0.0)))
            self.logger.record("custom/frag_delta", float(info.get("frag_delta", 0.0)))
            self.logger.record("custom/cluster_fragmentation_avg", float(info.get("cluster_fragmentation_avg", 0.0)))
            self.logger.record("custom/cluster_full_free_gpu_count", float(info.get("cluster_full_free_gpu_count", 0.0)))
            self.logger.record("custom/sigma_cpu_util", float(info.get("sigma_cpu_util", 0.0)))
            self.logger.record("custom/sigma_mem_util", float(info.get("sigma_mem_util", 0.0)))
            self.logger.record("custom/sigma_gpu_util", float(info.get("sigma_gpu_util", 0.0)))
            self.logger.record("custom/wait_time_ms", float(info.get("wait_time_ms", 0.0)))
            self.logger.record("custom/scheduled", 1.0 if bool(info.get("scheduled", False)) else 0.0)

            episode_metrics = info.get("episode_metrics")
            if isinstance(episode_metrics, dict):
                self.episode_counter += 1
                self.logger.record("episode/gar", float(episode_metrics.get("gar", 0.0)))
                self.logger.record("episode/gar_requested", float(episode_metrics.get("gar_requested", 0.0)))
                self.logger.record("episode/allocation_ratio_capacity", float(episode_metrics.get("allocation_ratio_capacity", 0.0)))
                self.logger.record("episode/unallocated_gpu_fraction", float(episode_metrics.get("unallocated_gpu_fraction", 0.0)))
                self.logger.record("episode/success_rate", float(episode_metrics.get("success_rate", 0.0)))
                self.logger.record("episode/mean_frag_delta", float(episode_metrics.get("mean_frag_delta", 0.0)))
                self.logger.record("episode/final_fragmentation_avg", float(episode_metrics.get("final_fragmentation_avg", 0.0)))
                self.logger.record("episode/sigma_cpu_util", float(episode_metrics.get("sigma_cpu_util", 0.0)))
                self.logger.record("episode/sigma_mem_util", float(episode_metrics.get("sigma_mem_util", 0.0)))
                self.logger.record("episode/sigma_gpu_util", float(episode_metrics.get("sigma_gpu_util", 0.0)))
                self.logger.record("episode/mean_wait_time_ms", float(episode_metrics.get("mean_wait_time_ms", 0.0)))
                self.logger.record("episode/reward", float(episode_metrics.get("episode_reward", 0.0)))
                self.logger.record("episode/count", float(self.episode_counter))
        return True


def make_env_factory(params: EnvParams) -> Callable[[], GPUSchedulingEnv]:
    def _factory() -> GPUSchedulingEnv:
        env_seed = params.seed + params.env_rank
        mode = "shuffle" if params.episode_order_mode == "shuffle" else "sequential"
        stride = 1 if mode == "shuffle" else params.n_envs
        env = build_env_from_preprocessed_csv(
            nodes_csv=params.nodes_csv,
            pods_csv=params.pods_csv,
            demand_distribution_json=params.demand_json,
            node_count=params.node_count,
            cpu_only_node_count=params.cpu_only_node_count,
            max_pods_per_episode=params.episode_len,
            seed=env_seed,
            episode_start_index=params.env_rank,
            episode_stride=stride,
            episode_order_mode=mode,
            fragmentation_mode=params.fragmentation_mode,
            fragmentation_scale=params.fragmentation_scale,
            frag_delta_scale=params.frag_delta_scale,
            gpu_capacity_scale=params.gpu_capacity_scale,
            frag_weight=params.frag_weight,
            balance_weight=params.balance_weight,
            util_weight=params.util_weight,
            global_frag_weight=params.global_frag_weight,
            free_gpu_weight=params.free_gpu_weight,
            free_gpu_penalty_mode=params.free_gpu_penalty_mode,
            success_reward=params.success_reward,
            fail_penalty=params.fail_penalty,
            slo_penalty=params.slo_penalty,
            slo_threshold_ms=params.slo_threshold_ms,
            priority_multiplier=params.priority_multiplier,
            sub_placement_policy=params.sub_placement_policy,
            gpu_only_pods=params.gpu_only_pods,
            pod_replication_factor=params.pod_replication_factor,
            min_pod_duration_ms=params.min_pod_duration_ms,
            invalid_action_mode="error",
        )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    return _factory


def _load_config_defaults(config_path: Path) -> Dict[str, object]:
    with config_path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"config JSON must be an object: {config_path}")
    return dict(payload)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _parse_int_csv(raw: str, *, field: str) -> List[int]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{field} cannot be empty")
    out: List[int] = []
    for token in vals:
        try:
            out.append(int(token))
        except ValueError as exc:
            raise ValueError(f"{field} must contain integers: {raw}") from exc
    return out


def _build_curriculum_schedule(args: argparse.Namespace) -> List[Tuple[int, int]]:
    if not args.use_curriculum:
        return [(int(args.pod_replication_factor), int(args.total_timesteps))]

    reps = _parse_int_csv(args.curriculum_replication_factors, field="curriculum_replication_factors")
    steps = _parse_int_csv(args.curriculum_phase_timesteps, field="curriculum_phase_timesteps")
    if len(reps) != len(steps):
        raise ValueError("curriculum_replication_factors and curriculum_phase_timesteps must have same length")
    if any(r < 1 for r in reps):
        raise ValueError("all curriculum replication factors must be >= 1")
    if any(s < 1 for s in steps):
        raise ValueError("all curriculum phase timesteps must be >= 1")

    total_from_phases = int(sum(steps))
    if total_from_phases != int(args.total_timesteps):
        print(
            f"Adjusting total_timesteps from {args.total_timesteps} to curriculum sum {total_from_phases}"
        )
        args.total_timesteps = total_from_phases
    return list(zip(reps, steps))


def _choose_dot_product_action(env: GPUSchedulingEnv, strict_mask: np.ndarray) -> int:
    feasible = [int(i) for i in np.flatnonzero(strict_mask)]
    if not feasible:
        return 0
    if env.sim is None:
        return feasible[0]

    pod = env._current_pod()
    best_idx = feasible[0]
    best_score: float | None = None
    for idx in feasible:
        node = env.sim.nodes[idx]
        node_vec = (
            node.cpu_avail,
            node.memory_avail,
            node.free_gpu_milli,
            max((g.free_milli for g in node.gpu_list), default=0),
        )
        pod_vec = (pod.cpu_milli, pod.memory_mib, pod.total_gpu_milli, pod.gpu_milli)
        score = float(sum(a * b for a, b in zip(pod_vec, node_vec)))
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _build_bc_dataset(
    *,
    args: argparse.Namespace,
    reward_cfg: Dict[str, float],
    replication_factor: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = build_env_from_preprocessed_csv(
        nodes_csv=args.nodes_csv,
        pods_csv=args.train_pods_csv,
        demand_distribution_json=args.demand_json,
        node_count=args.node_count,
        cpu_only_node_count=args.cpu_only_node_count,
        max_pods_per_episode=args.episode_len,
        seed=args.seed + 17,
        fragmentation_mode=args.fragmentation_mode,
        fragmentation_scale=args.fragmentation_scale,
        frag_delta_scale=args.frag_delta_scale,
        gpu_capacity_scale=args.gpu_capacity_scale,
        frag_weight=reward_cfg["frag_weight"],
        balance_weight=reward_cfg["balance_weight"],
        util_weight=reward_cfg["util_weight"],
        global_frag_weight=reward_cfg["global_frag_weight"],
        free_gpu_weight=reward_cfg["free_gpu_weight"],
        free_gpu_penalty_mode=reward_cfg["free_gpu_penalty_mode"],
        success_reward=reward_cfg["success_reward"],
        fail_penalty=reward_cfg["fail_penalty"],
        slo_penalty=reward_cfg["slo_penalty"],
        slo_threshold_ms=args.slo_threshold_ms,
        priority_multiplier=args.priority_multiplier,
        sub_placement_policy=args.sub_placement_policy,
        episode_order_mode="sequential",
        gpu_only_pods=args.gpu_only_pods,
        pod_replication_factor=replication_factor,
        min_pod_duration_ms=args.min_pod_duration_ms,
        invalid_action_mode="error",
    )
    obs, _ = env.reset(seed=args.seed + 19)

    max_steps = max(1, int(min(args.episode_len * args.bc_trace_fraction, args.bc_max_samples)))
    obs_list: List[np.ndarray] = []
    action_list: List[int] = []
    mask_list: List[np.ndarray] = []

    done = False
    steps = 0
    while not done and steps < max_steps:
        strict_mask = env.strict_action_mask().astype(bool)
        action = _choose_dot_product_action(env, strict_mask)
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        action_list.append(int(action))
        mask_list.append(strict_mask.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        steps += 1

    if not obs_list:
        return (
            np.zeros((0, env.obs_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, args.node_count), dtype=bool),
        )
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(action_list, dtype=np.int64),
        np.asarray(mask_list, dtype=bool),
    )


def _behavioral_clone_warmstart(
    *,
    model: MaskablePPO,
    obs_data: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    epochs: int,
    batch_size: int,
) -> None:
    if obs_data.shape[0] == 0:
        print("BC warm-start skipped: empty dataset")
        return

    policy = model.policy
    optimizer = policy.optimizer
    device = policy.device
    n = obs_data.shape[0]
    batch_size = max(1, int(batch_size))
    epochs = max(1, int(epochs))

    obs_t = torch.as_tensor(obs_data, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_obs = obs_t[idx]
            batch_actions = act_t[idx]
            batch_masks = mask_t[idx]

            distribution = policy.get_distribution(batch_obs, action_masks=batch_masks)
            log_prob = distribution.log_prob(batch_actions)
            loss = -log_prob.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(model.max_grad_norm))
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1
        print(f"BC epoch {epoch + 1}/{epochs}: n={n}, mean_nll={epoch_loss / max(1, batches):.6f}")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None, help="path to JSON config file")
    config_args, _ = config_parser.parse_known_args()
    config_defaults: Dict[str, object] = {}
    if config_args.config is not None:
        config_defaults = _load_config_defaults(config_args.config)

    parser = argparse.ArgumentParser(
        description="Phase 5 training with MaskablePPO",
        parents=[config_parser],
    )
    parser.add_argument("--nodes-csv", type=Path, default=Path("preprocessed/phase3/nodes_clean.csv"))
    parser.add_argument("--train-pods-csv", type=Path, default=Path("preprocessed/phase3/pods_test.csv"))
    parser.add_argument("--val-pods-csv", type=Path, default=Path("preprocessed/phase3/pods_test.csv"))
    parser.add_argument("--ablations", type=str, default="full", help="comma list: success_only,frag_only,full")
    parser.add_argument(
        "--demand-json",
        type=Path,
        default=Path("preprocessed/phase3/workload_demand_distribution.json"),
    )

    parser.add_argument("--node-count", type=int, default=64)
    parser.add_argument("--cpu-only-node-count", type=int, default=0)
    parser.add_argument("--episode-len", type=int, default=5000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--episode-order-mode",
        type=str,
        default="staggered",
        choices=["staggered", "shuffle"],
        help="staggered uses env offsets; shuffle samples random episodes",
    )
    parser.add_argument("--val-episode-len", type=int, default=0, help="<=0 means use full validation trace")

    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--patience-steps", type=int, default=100_000)

    parser.add_argument(
        "--fragmentation-mode",
        type=str,
        default="fgd",
        choices=["fgd", "utilization"],
        help="fgd uses demand-placeability; utilization uses used_gpu_ratio per node",
    )
    parser.add_argument("--fragmentation-scale", type=float, default=1.0)
    parser.add_argument("--frag-delta-scale", type=float, default=100.0)
    parser.add_argument("--gpu-capacity-scale", type=float, default=1.0)
    parser.add_argument("--frag-weight", type=float, default=120.0)
    parser.add_argument("--balance-weight", type=float, default=0.75)
    parser.add_argument("--util-weight", type=float, default=0.0)
    parser.add_argument("--global-frag-weight", type=float, default=12.0)
    parser.add_argument("--free-gpu-weight", type=float, default=0.03)
    parser.add_argument(
        "--free-gpu-penalty-mode",
        type=str,
        default="terminal",
        choices=["terminal", "step"],
        help="apply free-GPU penalty at episode end or every step",
    )
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--fail-penalty", type=float, default=-5.0)
    parser.add_argument("--slo-penalty", type=float, default=-2.0)
    parser.add_argument("--slo-threshold-ms", type=int, default=30_000)
    parser.add_argument("--priority-multiplier", type=float, default=3.0)
    parser.add_argument("--sub-placement-policy", type=str, default="most_used_first")
    parser.add_argument(
        "--gpu-only-pods",
        type=_parse_bool,
        default=True,
        help="if true, filter pods to num_gpu > 0 before building episodes",
    )
    parser.add_argument(
        "--pod-replication-factor",
        type=int,
        default=4,
        help="replicate the pod trace this many times to increase pressure",
    )
    parser.add_argument(
        "--min-pod-duration-ms",
        type=int,
        default=600_000,
        help="enforce minimum pod lifetime to create overlap and scheduling pressure",
    )

    parser.add_argument("--save-dir", type=Path, default=Path("models/phase5"))
    parser.add_argument("--tensorboard-log", type=Path, default=Path("runs/phase5"))
    parser.add_argument(
        "--norm-reward",
        type=_parse_bool,
        default=True,
        help="enable VecNormalize reward normalization",
    )
    parser.add_argument(
        "--policy-arch",
        type=str,
        default="attention",
        choices=["attention", "mlp"],
        help="policy backbone",
    )
    parser.add_argument("--attention-hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=2)

    parser.add_argument(
        "--use-curriculum",
        type=_parse_bool,
        default=True,
        help="train with increasing trace replication factors",
    )
    parser.add_argument(
        "--curriculum-replication-factors",
        type=str,
        default="2,3,4",
        help="comma-separated replication factors per phase",
    )
    parser.add_argument(
        "--curriculum-phase-timesteps",
        type=str,
        default="200000,200000,100000",
        help="comma-separated timesteps per curriculum phase",
    )
    parser.add_argument(
        "--validation-replication-factor",
        type=int,
        default=4,
        help="replication factor used for validation GAR",
    )

    parser.add_argument("--bc-warmstart", type=_parse_bool, default=True)
    parser.add_argument("--bc-policy", type=str, default="dot_product", choices=["dot_product"])
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-max-samples", type=int, default=20_000)
    parser.add_argument(
        "--bc-trace-fraction",
        type=float,
        default=0.5,
        help="fraction of one episode to use for BC dataset generation",
    )

    allowed_keys = {a.dest for a in parser._actions if a.dest != "help"}
    unknown_keys = sorted(k for k in config_defaults if k not in allowed_keys)
    if unknown_keys:
        raise ValueError(f"unknown config keys in {config_args.config}: {unknown_keys}")

    path_fields = {
        "config",
        "nodes_csv",
        "train_pods_csv",
        "val_pods_csv",
        "demand_json",
        "save_dir",
        "tensorboard_log",
    }
    for key in path_fields:
        if key in config_defaults and isinstance(config_defaults[key], str):
            config_defaults[key] = Path(config_defaults[key])

    if config_defaults:
        parser.set_defaults(**config_defaults)
    return parser.parse_args()


def build_ablation_rewards(base_args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    table: Dict[str, Dict[str, object]] = {
        "success_only": {
            "success_reward": base_args.success_reward,
            "fail_penalty": base_args.fail_penalty,
            "frag_weight": 0.0,
            "balance_weight": 0.0,
            "util_weight": 0.0,
            "global_frag_weight": 0.0,
            "free_gpu_weight": 0.0,
            "free_gpu_penalty_mode": base_args.free_gpu_penalty_mode,
            "slo_penalty": 0.0,
        },
        "frag_only": {
            "success_reward": base_args.success_reward,
            "fail_penalty": base_args.fail_penalty,
            "frag_weight": base_args.frag_weight,
            "balance_weight": 0.0,
            "util_weight": 0.0,
            "global_frag_weight": base_args.global_frag_weight,
            "free_gpu_weight": base_args.free_gpu_weight,
            "free_gpu_penalty_mode": base_args.free_gpu_penalty_mode,
            "slo_penalty": 0.0,
        },
        "full": {
            "success_reward": base_args.success_reward,
            "fail_penalty": base_args.fail_penalty,
            "frag_weight": base_args.frag_weight,
            "balance_weight": base_args.balance_weight,
            "util_weight": base_args.util_weight,
            "global_frag_weight": base_args.global_frag_weight,
            "free_gpu_weight": base_args.free_gpu_weight,
            "free_gpu_penalty_mode": base_args.free_gpu_penalty_mode,
            "slo_penalty": base_args.slo_penalty,
        },
    }
    wanted = [x.strip() for x in base_args.ablations.split(",") if x.strip()]
    invalid = [x for x in wanted if x not in table]
    if invalid:
        raise ValueError(f"unknown ablation names: {invalid}")
    return {k: table[k] for k in wanted}


def count_pods(csv_path: Path, *, gpu_only_pods: bool, pod_replication_factor: int) -> int:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if gpu_only_pods:
                num_gpu_raw = row.get("num_gpu", "0")
                if int(float(num_gpu_raw or 0.0)) <= 0:
                    continue
            count += 1
    factor = max(1, int(pod_replication_factor))
    return count * factor


def _build_train_vec_env(
    *,
    args: argparse.Namespace,
    reward_cfg: Dict[str, object],
    replication_factor: int,
) -> VecNormalize:
    train_fns: List[Callable[[], GPUSchedulingEnv]] = []
    for i in range(args.n_envs):
        train_fns.append(
            make_env_factory(
                EnvParams(
                    nodes_csv=args.nodes_csv,
                    pods_csv=args.train_pods_csv,
                    demand_json=args.demand_json,
                    node_count=args.node_count,
                    cpu_only_node_count=args.cpu_only_node_count,
                    episode_len=args.episode_len,
                    seed=args.seed,
                    env_rank=i,
                    n_envs=args.n_envs,
                    fragmentation_mode=args.fragmentation_mode,
                    fragmentation_scale=args.fragmentation_scale,
                    frag_delta_scale=args.frag_delta_scale,
                    gpu_capacity_scale=args.gpu_capacity_scale,
                    frag_weight=float(reward_cfg["frag_weight"]),
                    balance_weight=float(reward_cfg["balance_weight"]),
                    util_weight=float(reward_cfg["util_weight"]),
                    global_frag_weight=float(reward_cfg["global_frag_weight"]),
                    free_gpu_weight=float(reward_cfg["free_gpu_weight"]),
                    free_gpu_penalty_mode=str(reward_cfg["free_gpu_penalty_mode"]),
                    success_reward=float(reward_cfg["success_reward"]),
                    fail_penalty=float(reward_cfg["fail_penalty"]),
                    slo_penalty=float(reward_cfg["slo_penalty"]),
                    slo_threshold_ms=args.slo_threshold_ms,
                    priority_multiplier=args.priority_multiplier,
                    sub_placement_policy=args.sub_placement_policy,
                    episode_order_mode=args.episode_order_mode,
                    gpu_only_pods=args.gpu_only_pods,
                    pod_replication_factor=replication_factor,
                    min_pod_duration_ms=args.min_pod_duration_ms,
                )
            )
        )

    vec_train = SubprocVecEnv(train_fns) if args.n_envs > 1 else DummyVecEnv(train_fns)
    return VecNormalize(vec_train, norm_obs=True, norm_reward=args.norm_reward, clip_obs=10.0)


def _build_eval_vec_env(
    *,
    args: argparse.Namespace,
    reward_cfg: Dict[str, object],
) -> VecNormalize:
    if args.val_episode_len > 0:
        val_episode_len = args.val_episode_len
    else:
        val_episode_len = count_pods(
            args.val_pods_csv,
            gpu_only_pods=args.gpu_only_pods,
            pod_replication_factor=args.validation_replication_factor,
        )
        if val_episode_len <= 0:
            raise ValueError(f"validation pod file has no data: {args.val_pods_csv}")

    eval_fn = make_env_factory(
        EnvParams(
            nodes_csv=args.nodes_csv,
            pods_csv=args.val_pods_csv,
            demand_json=args.demand_json,
            node_count=args.node_count,
            cpu_only_node_count=args.cpu_only_node_count,
            episode_len=val_episode_len,
            seed=args.seed + 999,
            env_rank=0,
            n_envs=1,
            fragmentation_mode=args.fragmentation_mode,
            fragmentation_scale=args.fragmentation_scale,
            frag_delta_scale=args.frag_delta_scale,
            gpu_capacity_scale=args.gpu_capacity_scale,
            frag_weight=float(reward_cfg["frag_weight"]),
            balance_weight=float(reward_cfg["balance_weight"]),
            util_weight=float(reward_cfg["util_weight"]),
            global_frag_weight=float(reward_cfg["global_frag_weight"]),
            free_gpu_weight=float(reward_cfg["free_gpu_weight"]),
            free_gpu_penalty_mode=str(reward_cfg["free_gpu_penalty_mode"]),
            success_reward=float(reward_cfg["success_reward"]),
            fail_penalty=float(reward_cfg["fail_penalty"]),
            slo_penalty=float(reward_cfg["slo_penalty"]),
            slo_threshold_ms=args.slo_threshold_ms,
            priority_multiplier=args.priority_multiplier,
            sub_placement_policy=args.sub_placement_policy,
            episode_order_mode="sequential",
            gpu_only_pods=args.gpu_only_pods,
            pod_replication_factor=args.validation_replication_factor,
            min_pod_duration_ms=args.min_pod_duration_ms,
        )
    )
    vec_eval = DummyVecEnv([eval_fn])
    vec_eval = VecNormalize(vec_eval, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec_eval.training = False
    return vec_eval


def train_one_experiment(
    *,
    args: argparse.Namespace,
    device: str,
    name: str,
    reward_cfg: Dict[str, object],
) -> Dict[str, object]:
    save_dir = args.save_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = args.tensorboard_log / name
    tb_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / "train_config.json").open("w") as f:
        json.dump({
            "experiment": name,
            "reward_config": reward_cfg,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "device": device,
        }, f, indent=2)

    curriculum = _build_curriculum_schedule(args)
    vec_train = _build_train_vec_env(
        args=args,
        reward_cfg=reward_cfg,
        replication_factor=curriculum[0][0],
    )
    vec_eval = _build_eval_vec_env(args=args, reward_cfg=reward_cfg)
    vec_eval.obs_rms = vec_train.obs_rms

    if args.policy_arch == "attention":
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=nn.ReLU,
            features_extractor_class=NodeAttentionExtractor,
            features_extractor_kwargs=dict(
                node_count=args.node_count,
                node_feature_dim=8,
                pod_feature_dim=17,
                global_feature_dim=10,
                hidden_dim=args.attention_hidden_dim,
                num_heads=args.attention_heads,
                num_attn_layers=args.attention_layers,
            ),
        )
    else:
        policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_train,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=str(tb_dir),
        verbose=1,
        seed=args.seed,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    if args.bc_warmstart:
        bc_replication = curriculum[-1][0]
        bc_obs, bc_actions, bc_masks = _build_bc_dataset(
            args=args,
            reward_cfg=reward_cfg,
            replication_factor=bc_replication,
        )
        _behavioral_clone_warmstart(
            model=model,
            obs_data=bc_obs,
            actions=bc_actions,
            masks=bc_masks,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
        )

    metrics_callback = SchedulingMetricsCallback(verbose=0)
    validation_callback = ValidationGARCallback(
        eval_env=vec_eval,
        eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
        n_eval_episodes=args.eval_episodes,
        best_model_dir=save_dir,
        patience_steps=args.patience_steps,
        verbose=1,
    )

    for phase_idx, (replication_factor, phase_steps) in enumerate(curriculum):
        if phase_idx > 0:
            prev_env = model.get_env()
            new_train_env = _build_train_vec_env(
                args=args,
                reward_cfg=reward_cfg,
                replication_factor=replication_factor,
            )
            if isinstance(prev_env, VecNormalize) and isinstance(new_train_env, VecNormalize):
                new_train_env.obs_rms = prev_env.obs_rms
                if args.norm_reward:
                    new_train_env.ret_rms = prev_env.ret_rms
            model.set_env(new_train_env)
            if isinstance(prev_env, VecNormalize):
                prev_env.close()
            vec_train = new_train_env
            vec_eval.obs_rms = vec_train.obs_rms

        print(
            f"Training phase {phase_idx + 1}/{len(curriculum)}: "
            f"replication={replication_factor}, timesteps={phase_steps}"
        )
        model.learn(
            total_timesteps=phase_steps,
            callback=[validation_callback, metrics_callback],
            progress_bar=True,
            reset_num_timesteps=(phase_idx == 0),
            tb_log_name=f"{name}_phase{phase_idx + 1}",
        )

    last_model_path = save_dir / "last_model"
    model.save(str(last_model_path))
    vec_train.save(str(save_dir / "vecnormalize.pkl"))

    result = {
        "experiment": name,
        "best_model_path": str(save_dir / "best_model"),
        "best_vecnormalize_path": str(save_dir / "best_vecnormalize.pkl"),
        "best_model_stats_path": str(save_dir / "best_model.pkl"),
        "last_model_path": str(last_model_path),
        "vecnormalize_path": str(save_dir / "vecnormalize.pkl"),
        "reward_config": reward_cfg,
        "curriculum": [
            {"replication_factor": int(rep), "timesteps": int(steps)}
            for rep, steps in curriculum
        ],
        "policy_arch": args.policy_arch,
        "bc_warmstart": bool(args.bc_warmstart),
        "device": device,
        "best_gar": validation_callback.best_gar,
        "best_step": validation_callback.best_step,
    }

    vec_train.close()
    vec_eval.close()
    return result


def main() -> None:  
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    reward_experiments = build_ablation_rewards(args)
    all_results: List[Dict[str, object]] = []

    for name, reward_cfg in reward_experiments.items():
        print(f"\n=== Training experiment: {name} ===")
        result = train_one_experiment(
            args=args,
            device=device,
            name=name,
            reward_cfg=reward_cfg,
        )
        all_results.append(result)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.save_dir / "training_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
