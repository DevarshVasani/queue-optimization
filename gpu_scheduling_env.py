from __future__ import annotations

import copy
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym  # type: ignore[no-redef]
        from gym import spaces  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        class _BaseEnv:
            metadata: Dict[str, object] = {}

        class _Discrete:
            def __init__(self, n: int) -> None:
                self.n = int(n)

        class _Box:
            def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: object) -> None:
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _FallbackSpaces:
            Discrete = _Discrete
            Box = _Box

        class _FallbackGym:
            Env = _BaseEnv

        gym = _FallbackGym()  # type: ignore[assignment]
        spaces = _FallbackSpaces()  # type: ignore[assignment]

from cluster_sim import ClusterSimulator, Node, Pod, load_nodes


GPU_MODEL_ORDER = ["A10", "G2", "G3", "P100", "T4", "V100M16", "V100M32"]
NODE_MODEL_ORDER = ["CPU-only", *GPU_MODEL_ORDER]
QOS_ORDER = ["Guaranteed", "Burstable", "BestEffort", "LS"]
DEFAULT_MIN_POD_DURATION_MS = 600_000


def _normalize_qos(raw_qos: str) -> str:
    s = raw_qos.strip()
    if not s:
        return "BestEffort"
    lower = s.lower()
    if lower == "guaranteed":
        return "Guaranteed"
    if lower == "burstable":
        return "Burstable"
    if lower in {"be", "besteffort", "best-effort", "best_effort"}:
        return "BestEffort"
    if lower == "ls":
        return "LS"
    return "BestEffort"


def _priority_from_pod(pod: Pod) -> int:
    return 1 if int(pod.priority) > 0 else 0


def _parse_optional_int(raw: object) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        raw = s
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _coerce_deletion_time(
    creation_time: int,
    deletion_time_raw: object,
    *,
    min_pod_duration_ms: int,
) -> int:
    min_duration = max(0, int(min_pod_duration_ms))
    deletion_time = _parse_optional_int(deletion_time_raw)
    if deletion_time is None:
        return creation_time + max(1, min_duration)

    min_valid_deletion = creation_time + max(1, min_duration)
    if deletion_time < min_valid_deletion:
        return min_valid_deletion
    return deletion_time


def _clone_pod_with_min_duration(pod: Pod, *, min_pod_duration_ms: int) -> Pod:
    p = copy.deepcopy(pod)
    p.deletion_time = _coerce_deletion_time(
        int(p.creation_time),
        p.deletion_time,
        min_pod_duration_ms=min_pod_duration_ms,
    )
    p.scheduled_time = None
    return p


def _pod_from_mapping(row: Dict[str, object], idx: int, *, min_pod_duration_ms: int) -> Pod:
    name = str(row.get("name", f"pod-{idx}"))
    cpu_milli = int(float(row.get("cpu_milli", 0)))
    memory_mib = int(float(row.get("memory_mib", 0)))
    num_gpu = int(float(row.get("num_gpu", 0)))
    gpu_milli = int(float(row.get("gpu_milli", 0)))
    raw_spec = str(row.get("gpu_spec", "")).strip()
    if raw_spec and raw_spec.lower() not in {"nan", "none", "null"}:
        gpu_spec = tuple(x.strip() for x in raw_spec.split("|") if x.strip())
    else:
        gpu_spec = ()
    qos = _normalize_qos(str(row.get("qos", "")))
    creation_time = int(float(row.get("creation_time", 0)))
    deletion_time = _coerce_deletion_time(
        creation_time,
        row.get("deletion_time"),
        min_pod_duration_ms=min_pod_duration_ms,
    )

    priority_raw = row.get("priority", None)
    if priority_raw is None:
        priority = 1 if (qos == "Guaranteed" and 0 < gpu_milli < 1000) else 0
    else:
        priority = int(float(priority_raw))

    return Pod(
        name=name,
        cpu_milli=cpu_milli,
        memory_mib=memory_mib,
        num_gpu=num_gpu,
        gpu_milli=gpu_milli,
        gpu_spec=gpu_spec,
        qos=qos,
        priority=priority,
        creation_time=creation_time,
        deletion_time=deletion_time,
        scheduled_time=None,
    )


def _node_from_mapping(row: Dict[str, object], idx: int) -> Node:
    node_name = str(row.get("sn", row.get("node_id", f"node-{idx}"))).strip()
    if not node_name:
        node_name = f"node-{idx}"
    cpu_total = int(float(row.get("cpu_milli", 0)))
    mem_total = int(float(row.get("memory_mib", 0)))
    gpu_count = int(float(row.get("gpu", 0)))
    model = str(row.get("model", "CPU-only")).strip() or "CPU-only"
    if gpu_count <= 0:
        model = "CPU-only"
        gpu_list = []
    else:
        from cluster_sim import GPU

        gpu_list = [GPU(allocated_milli=0, gpu_type=model) for _ in range(gpu_count)]
    return Node(
        node_id=node_name,
        cpu_total=cpu_total,
        memory_total=mem_total,
        model=model,
        gpu_list=gpu_list,
    )


@dataclass
class RewardBreakdown:
    success: float
    fragmentation: float
    global_fragmentation: float
    balance: float
    free_gpu: float
    utilization: float
    slo: float

    @property
    def total(self) -> float:
        return (
            self.success
            + self.fragmentation
            + self.global_fragmentation
            + self.balance
            + self.free_gpu
            + self.utilization
            + self.slo
        )


class GPUSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        nodes: Sequence[Union[Node, Dict[str, object]]],
        pods: Sequence[Union[Pod, Dict[str, object]]],
        demand_distribution: Dict[str, Dict[Tuple[int, int], float]],
        *,
        fragmentation_mode: str = "fgd",
        fragmentation_scale: float = 1.0,
        frag_delta_scale: float = 100.0,
        frag_weight: float = 2.0,
        balance_weight: float = 0.5,
        util_weight: float = 0.0,
        global_frag_weight: float = 0.0,
        free_gpu_weight: float = 0.0,
        free_gpu_penalty_mode: str = "terminal",
        slo_threshold_ms: int = 30_000,
        slo_penalty: float = -2.0,
        priority_multiplier: float = 3.0,
        success_reward: float = 1.0,
        fail_penalty: float = -5.0,
        max_pods_per_episode: int = 500,
        node_count: int = 128,
        cpu_only_node_count: int = 16,
        sub_placement_policy: str = "most_used_first",
        gpu_capacity_scale: float = 1.0,
        episode_start_index: int = 0,
        episode_stride: int = 1,
        episode_order_mode: str = "sequential",
        invalid_action_mode: str = "error",
        reward_clip: Optional[Tuple[float, float]] = None,
        min_pod_duration_ms: int = DEFAULT_MIN_POD_DURATION_MS,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.fragmentation_mode = fragmentation_mode
        self.fragmentation_scale = float(fragmentation_scale)
        self.frag_delta_scale = float(frag_delta_scale)
        self.frag_weight = float(frag_weight)
        self.balance_weight = float(balance_weight)
        self.util_weight = float(util_weight)
        self.global_frag_weight = float(global_frag_weight)
        self.free_gpu_weight = float(free_gpu_weight)
        self.free_gpu_penalty_mode = str(free_gpu_penalty_mode).strip().lower()
        if self.free_gpu_penalty_mode not in {"terminal", "step"}:
            raise ValueError("free_gpu_penalty_mode must be 'terminal' or 'step'")
        self.slo_threshold_ms = int(slo_threshold_ms)
        self.slo_penalty = float(slo_penalty)
        self.priority_multiplier = float(priority_multiplier)
        self.success_reward = float(success_reward)
        self.fail_penalty = float(fail_penalty)
        self.max_pods_per_episode = int(max_pods_per_episode)
        self.node_count = int(node_count)
        self.cpu_only_node_count = int(max(0, cpu_only_node_count))
        self.sub_placement_policy = sub_placement_policy
        self.gpu_capacity_scale = float(np.clip(gpu_capacity_scale, 0.05, 1.0))
        self.episode_start_index = int(max(0, episode_start_index))
        self.episode_stride = int(max(1, episode_stride))
        self.episode_order_mode = episode_order_mode.strip().lower()
        if self.episode_order_mode not in {"sequential", "shuffle"}:
            raise ValueError("episode_order_mode must be 'sequential' or 'shuffle'")
        self.invalid_action_mode = invalid_action_mode
        self.reward_clip = reward_clip
        self.min_pod_duration_ms = int(max(0, min_pod_duration_ms))

        self._seed_value = seed
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)

        self._all_nodes = [n if isinstance(n, Node) else _node_from_mapping(n, i) for i, n in enumerate(nodes)]
        self._all_pods = [
            _clone_pod_with_min_duration(p, min_pod_duration_ms=self.min_pod_duration_ms)
            if isinstance(p, Pod)
            else _pod_from_mapping(p, i, min_pod_duration_ms=self.min_pod_duration_ms)
            for i, p in enumerate(pods)
        ]
        if not self._all_nodes:
            raise ValueError("nodes cannot be empty")
        if not self._all_pods:
            raise ValueError("pods cannot be empty")

        self.max_cpu_for_norm = max(1, max(n.cpu_total for n in self._all_nodes))
        self.max_mem_for_norm = max(1, max(n.memory_total for n in self._all_nodes))
        self.max_gpu_per_node = max(1, max(n.gpu_count for n in self._all_nodes))
        self.trace_max_timestamp = max(1, max(p.creation_time for p in self._all_pods))

        self.nodes_template = self._select_nodes(self._all_nodes)
        if len(self.nodes_template) != self.node_count:
            raise RuntimeError("internal error: selected nodes length != node_count")
        scaled_per_gpu_capacity = int(round(1000 * self.gpu_capacity_scale))
        self.cluster_total_gpu_capacity_milli = sum(n.gpu_count * scaled_per_gpu_capacity for n in self.nodes_template)

        self.demand_distribution = demand_distribution
        self.episodes = self._build_episodes(self._all_pods)
        if not self.episodes:
            raise ValueError("no full episodes available; increase pod count or lower max_pods_per_episode")

        self.current_episode_number = -1
        self._episodes_served = 0
        self._episode_pods: List[Pod] = []
        self._pod_idx = 0
        self._terminated = False
        self._current_mask = np.zeros((self.node_count,), dtype=bool)
        self._strict_mask = np.zeros((self.node_count,), dtype=bool)
        self._synced_pod_idx = -1
        self._episode_requested_gpu_milli = 0
        self._episode_allocated_gpu_milli = 0
        self._episode_scheduled = 0
        self._episode_failed = 0
        self._episode_frag_delta_sum = 0.0
        self._episode_wait_ms_sum = 0.0
        self._episode_reward_sum = 0.0
        self._episode_live_gpu_milli_sum = 0.0
        self._episode_live_gpu_peak_milli = 0.0
        self._episode_live_gpu_last_milli = 0.0

        self.sim: Optional[ClusterSimulator] = None

        self.pod_feature_dim = 17
        self.node_feature_dim = 8
        self.global_feature_dim = len(GPU_MODEL_ORDER) + 3
        self.obs_dim = self.pod_feature_dim + self.node_count * self.node_feature_dim + self.global_feature_dim

        self.action_space = spaces.Discrete(self.node_count)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self._seed_value = seed
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
        return [] if seed is None else [seed]

    def _select_nodes(self, nodes: Sequence[Node]) -> List[Node]:
        gpu_nodes = [n for n in nodes if n.gpu_count > 0]
        cpu_nodes = [n for n in nodes if n.gpu_count == 0]

        gpu_nodes_sorted = sorted(gpu_nodes, key=lambda n: (-n.gpu_count, n.node_id))
        cpu_nodes_sorted = sorted(cpu_nodes, key=lambda n: n.node_id)

        gpu_budget = max(0, self.node_count - self.cpu_only_node_count)
        selected: List[Node] = []
        selected.extend(gpu_nodes_sorted[:gpu_budget])

        remaining = self.node_count - len(selected)
        if remaining > 0:
            selected.extend(cpu_nodes_sorted[:remaining])

        if len(selected) < self.node_count:
            leftovers = [n for n in gpu_nodes_sorted[gpu_budget:] if n not in selected]
            selected.extend(leftovers[: self.node_count - len(selected)])

        if len(selected) != self.node_count:
            raise ValueError(
                f"unable to select node_count={self.node_count} from available={len(nodes)}"
            )
        return sorted(selected, key=lambda n: n.node_id)

    def _build_episodes(self, pods: Sequence[Pod]) -> List[Tuple[int, int]]:
        ordered = sorted(pods, key=lambda p: (p.creation_time, p.name))
        usable = len(ordered) - (len(ordered) % self.max_pods_per_episode)
        out: List[Tuple[int, int]] = []
        for start in range(0, usable, self.max_pods_per_episode):
            out.append((start, start + self.max_pods_per_episode))
        self._ordered_pods = ordered
        return out

    def _qos_onehot(self, qos: str) -> List[float]:
        q = _normalize_qos(qos)
        return [1.0 if q == name else 0.0 for name in QOS_ORDER]

    def _pod_model_onehot(self, pod: Pod) -> List[float]:
        out = [0.0] * (1 + len(GPU_MODEL_ORDER))
        if not pod.gpu_spec:
            out[0] = 1.0
            return out
        for i, model in enumerate(GPU_MODEL_ORDER, start=1):
            if model in pod.gpu_spec:
                out[i] = 1.0
        return out

    def _node_model_scalar(self, model: str) -> float:
        if model not in NODE_MODEL_ORDER:
            return 0.0
        idx = NODE_MODEL_ORDER.index(model)
        return idx / max(1, len(NODE_MODEL_ORDER) - 1)

    def _current_pod(self) -> Pod:
        return self._episode_pods[self._pod_idx]

    def _compute_mask_for_pod(self, pod: Pod) -> np.ndarray:
        if self.sim is None:
            raise RuntimeError("simulator is not initialised")
        mask = np.zeros((self.node_count,), dtype=bool)
        for i in range(self.node_count):
            mask[i] = self.sim.check_feasibility(pod, i)
        return mask

    def _safe_mask(self, strict_mask: np.ndarray) -> np.ndarray:
        if np.any(strict_mask):
            return strict_mask
        return np.ones_like(strict_mask, dtype=bool)

    def _sync_current_pod(self) -> None:
        if self._terminated:
            return
        if self._pod_idx == self._synced_pod_idx:
            return
        if self.sim is None:
            raise RuntimeError("simulator is not initialised")

        pod = self._current_pod()
        self.sim.advance_to_time(pod.creation_time)
        strict_mask = self._compute_mask_for_pod(pod)
        self._strict_mask = strict_mask
        self._current_mask = self._safe_mask(strict_mask)
        self._synced_pod_idx = self._pod_idx

    def _node_balance_penalty(self) -> float:
        sigma_cpu, sigma_mem, sigma_gpu = self._utilization_sigmas()
        return -self.balance_weight * (sigma_cpu + sigma_mem + sigma_gpu)

    def _utilization_sigmas(self) -> Tuple[float, float, float]:
        if self.sim is None:
            raise RuntimeError("simulator is not initialised")

        cpu_utils: List[float] = []
        mem_utils: List[float] = []
        gpu_utils: List[float] = []
        for node in self.sim.nodes:
            cpu_used = node.cpu_total - node.cpu_avail
            mem_used = node.memory_total - node.memory_avail
            cpu_utils.append(cpu_used / max(1, node.cpu_total))
            mem_utils.append(mem_used / max(1, node.memory_total))
            if node.gpu_count > 0:
                gpu_utils.append(node.used_gpu_milli / max(1, node.gpu_count * 1000))
            else:
                gpu_utils.append(0.0)

        sigma_cpu = float(np.std(np.asarray(cpu_utils, dtype=np.float32)))
        sigma_mem = float(np.std(np.asarray(mem_utils, dtype=np.float32)))
        sigma_gpu = float(np.std(np.asarray(gpu_utils, dtype=np.float32)))
        return sigma_cpu, sigma_mem, sigma_gpu

    def _cluster_fragmentation_average(self) -> float:
        if self.sim is None:
            raise RuntimeError("simulator is not initialised")
        return float(np.mean([self.sim.get_fragmentation_score(i) for i in range(self.node_count)]))

    def _cluster_full_free_gpu_count(self) -> float:
        if self.sim is None:
            raise RuntimeError("simulator is not initialised")
        return float(sum(1 for node in self.sim.nodes for gpu in node.gpu_list if gpu.allocated_milli == 0))

    def _compute_reward(
        self,
        *,
        pod: Pod,
        scheduled: bool,
        no_feasible: bool,
        frag_before: float,
        frag_after: float,
        terminated: bool,
    ) -> RewardBreakdown:
        if no_feasible or not scheduled:
            return RewardBreakdown(
                success=self.fail_penalty,
                fragmentation=0.0,
                global_fragmentation=0.0,
                balance=0.0,
                free_gpu=0.0,
                utilization=0.0,
                slo=0.0,
            )

        delta = max(0.0, frag_after - frag_before) * self.frag_delta_scale
        r_success = self.success_reward
        r_frag = -self.frag_weight * delta
        r_global_frag = -self.global_frag_weight * self._cluster_fragmentation_average()
        r_bal = self._node_balance_penalty()
        r_util = self.util_weight * (float(pod.total_gpu_milli) / 1000.0)
        apply_free_gpu_penalty = self.free_gpu_penalty_mode == "step" or (
            self.free_gpu_penalty_mode == "terminal" and terminated
        )
        r_free_gpu = (
            -self.free_gpu_weight * self._cluster_full_free_gpu_count()
            if apply_free_gpu_penalty
            else 0.0
        )

        wait = 0
        if self.sim is not None:
            wait = max(0, int(self.sim.current_time) - int(pod.creation_time))
        if wait > self.slo_threshold_ms:
            scale = self.priority_multiplier if _priority_from_pod(pod) > 0 else 1.0
            r_slo = self.slo_penalty * scale
        else:
            r_slo = 0.0

        return RewardBreakdown(
            success=r_success,
            fragmentation=r_frag,
            global_fragmentation=r_global_frag,
            balance=r_bal,
            free_gpu=r_free_gpu,
            utilization=r_util,
            slo=r_slo,
        )

    def _pod_features(self, pod: Pod) -> List[float]:
        out = [
            float(pod.num_gpu) / float(self.max_gpu_per_node),
            float(pod.gpu_milli) / 1000.0,
            float(pod.cpu_milli) / float(self.max_cpu_for_norm),
            float(pod.memory_mib) / float(self.max_mem_for_norm),
        ]
        out.extend(self._pod_model_onehot(pod))
        out.extend(self._qos_onehot(pod.qos))
        out.append(float(_priority_from_pod(pod)))
        return out

    def _node_features(self) -> List[float]:
        if self.sim is None:
            raise RuntimeError("simulator is not initialised")
        out: List[float] = []
        for idx, node in enumerate(self.sim.nodes):
            cpu_avail_ratio = node.cpu_avail / max(1, node.cpu_total)
            mem_avail_ratio = node.memory_avail / max(1, node.memory_total)
            gpu_avail_count = sum(1 for g in node.gpu_list if g.allocated_milli < 1000)
            gpu_avail_ratio = gpu_avail_count / float(self.max_gpu_per_node)
            model_scalar = self._node_model_scalar(node.model if node.gpu_count > 0 else "CPU-only")
            frag_score = self.sim.get_fragmentation_score(idx)
            max_free_milli_segment = max((g.free_milli for g in node.gpu_list), default=0) / 1000.0
            full_free_gpus = sum(1 for g in node.gpu_list if g.allocated_milli == 0)
            full_free_ratio = full_free_gpus / float(self.max_gpu_per_node)
            partial_gpus = sum(1 for g in node.gpu_list if 0 < g.allocated_milli < 1000)
            partial_gpu_ratio = partial_gpus / float(self.max_gpu_per_node)
            out.extend(
                [
                    float(cpu_avail_ratio),
                    float(mem_avail_ratio),
                    float(gpu_avail_ratio),
                    float(model_scalar),
                    float(frag_score),
                    float(max_free_milli_segment),
                    float(full_free_ratio),
                    float(partial_gpu_ratio),
                ]
            )
        return out

    def _global_features(self) -> List[float]:
        if self.sim is None:
            raise RuntimeError("simulator is not initialised")
        per_model_free: List[float] = []
        denom = float(self.node_count * self.max_gpu_per_node)

        for model in GPU_MODEL_ORDER:
            free_gpu_count = 0
            for node in self.sim.nodes:
                if node.model != model or node.gpu_count <= 0:
                    continue
                free_gpu_count += sum(1 for g in node.gpu_list if g.allocated_milli == 0)
            per_model_free.append(free_gpu_count / denom)

        frag_avg = self._cluster_fragmentation_average()
        pods_pending = 0.0
        current_time_norm = float(self.sim.current_time) / float(self.trace_max_timestamp)

        return [*per_model_free, frag_avg, pods_pending, current_time_norm]

    def _get_obs(self) -> np.ndarray:
        if self._terminated:
            return np.zeros((self.obs_dim,), dtype=np.float32)

        pod = self._current_pod()
        vec = self._pod_features(pod)
        vec.extend(self._node_features())
        vec.extend(self._global_features())

        if len(vec) != self.obs_dim:
            raise RuntimeError(f"observation length mismatch: got={len(vec)} expected={self.obs_dim}")
        return np.asarray(vec, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        self._sync_current_pod()
        return self._current_mask.copy()

    def valid_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def strict_action_mask(self) -> np.ndarray:
        self._sync_current_pod()
        return self._strict_mask.copy()

    def _build_info(self, *, pod: Pod, reward: RewardBreakdown, scheduled: bool, reason: str, action: int) -> Dict[str, object]:
        sigma_cpu, sigma_mem, sigma_gpu = self._utilization_sigmas()
        wait_time_ms = 0
        if self.sim is not None:
            wait_time_ms = max(0, int(self.sim.current_time) - int(pod.creation_time))
        return {
            "pod_name": pod.name,
            "pod_index": self._pod_idx,
            "pod_total_gpu_milli": pod.total_gpu_milli,
            "scheduled": scheduled,
            "reason": reason,
            "action": action,
            "strict_action_mask": self._strict_mask.astype(np.int8),
            "action_mask": self._current_mask.astype(np.int8),
            "reward_success": reward.success,
            "reward_fragmentation": reward.fragmentation,
            "reward_global_fragmentation": reward.global_fragmentation,
            "reward_balance": reward.balance,
            "reward_free_gpu": reward.free_gpu,
            "reward_utilization": reward.utilization,
            "reward_slo": reward.slo,
            "reward_total": reward.total,
            "sigma_cpu_util": sigma_cpu,
            "sigma_mem_util": sigma_mem,
            "sigma_gpu_util": sigma_gpu,
            "wait_time_ms": float(wait_time_ms),
            "cluster_fragmentation_avg": self._cluster_fragmentation_average(),
            "cluster_full_free_gpu_count": self._cluster_full_free_gpu_count(),
            "current_time": 0 if self.sim is None else self.sim.current_time,
            "episode_requested_gpu_milli": self._episode_requested_gpu_milli,
            "episode_allocated_gpu_milli": self._episode_allocated_gpu_milli,
            "episode_scheduled": self._episode_scheduled,
            "episode_failed": self._episode_failed,
        }

    def get_metrics(self) -> Dict[str, float]:
        requested = float(self._episode_requested_gpu_milli)
        allocated = float(self._episode_allocated_gpu_milli)
        cap = float(self.cluster_total_gpu_capacity_milli)
        steps = float(max(1, self._episode_scheduled + self._episode_failed))
        allocation_ratio_requested = 0.0 if requested <= 0 else allocated / requested
        allocation_ratio_capacity = 0.0 if cap <= 0 else self._episode_live_gpu_milli_sum / (steps * cap)
        allocation_ratio_capacity_peak = 0.0 if cap <= 0 else self._episode_live_gpu_peak_milli / cap
        allocation_ratio_capacity_final = 0.0 if cap <= 0 else self._episode_live_gpu_last_milli / cap
        allocation_ratio_capacity_cumulative = 0.0 if cap <= 0 else allocated / cap
        sigma_cpu, sigma_mem, sigma_gpu = self._utilization_sigmas()
        frag_avg = self._cluster_fragmentation_average()
        full_free_gpu_count = self._cluster_full_free_gpu_count()
        return {
            "requested_gpu_milli": requested,
            "allocated_gpu_milli": allocated,
            "gpu_capacity_milli": cap,
            "allocation_ratio_capacity": allocation_ratio_capacity,
            "allocation_ratio_capacity_peak": allocation_ratio_capacity_peak,
            "allocation_ratio_capacity_final": allocation_ratio_capacity_final,
            "allocation_ratio_capacity_cumulative": allocation_ratio_capacity_cumulative,
            "allocation_ratio_requested": allocation_ratio_requested,
            "gar": allocation_ratio_capacity,
            "gar_requested": allocation_ratio_requested,
            "unallocated_gpu_fraction": float(np.clip(1.0 - allocation_ratio_capacity, 0.0, 1.0)),
            "success_rate": self._episode_scheduled / steps,
            "mean_frag_delta": self._episode_frag_delta_sum / steps,
            "mean_wait_time_ms": self._episode_wait_ms_sum / steps,
            "final_fragmentation_avg": frag_avg,
            "full_free_gpu_count": full_free_gpu_count,
            "sigma_cpu_util": sigma_cpu,
            "sigma_mem_util": sigma_mem,
            "sigma_gpu_util": sigma_gpu,
            "episode_reward": self._episode_reward_sum,
            "episode_steps": steps,
        }

    def get_total_gpu_capacity_milli(self) -> int:
        return int(self.cluster_total_gpu_capacity_milli)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        if seed is not None:
            self.seed(seed)

        if options and "episode_index" in options:
            episode_idx = int(options["episode_index"]) % len(self.episodes)
        elif self.episode_order_mode == "shuffle":
            episode_idx = int(self._rng.integers(0, len(self.episodes)))
        else:
            episode_idx = (
                self.episode_start_index + self._episodes_served * self.episode_stride
            ) % len(self.episodes)

        start, end = self.episodes[episode_idx]
        episode_pods = self._ordered_pods[start:end]
        self.sim = ClusterSimulator(
            nodes=self.nodes_template,
            pods=episode_pods,
            demand_distribution=self.demand_distribution,
            fragmentation_mode=self.fragmentation_mode,
            fragmentation_scale=self.fragmentation_scale,
            record_history=False,
        )
        self.sim.reset()
        if self.gpu_capacity_scale < 1.0:
            baseline_reserved = int(round((1.0 - self.gpu_capacity_scale) * 1000))
            baseline_reserved = int(np.clip(baseline_reserved, 0, 950))
            if baseline_reserved > 0:
                for node in self.sim.nodes:
                    for gpu in node.gpu_list:
                        gpu.allocated_milli = baseline_reserved
        self._episode_pods = self.sim.pods
        self._pod_idx = 0
        self._terminated = False
        self._synced_pod_idx = -1
        self._episode_requested_gpu_milli = 0
        self._episode_allocated_gpu_milli = 0
        self._episode_scheduled = 0
        self._episode_failed = 0
        self._episode_frag_delta_sum = 0.0
        self._episode_wait_ms_sum = 0.0
        self._episode_reward_sum = 0.0
        self._episode_live_gpu_milli_sum = 0.0
        self._episode_live_gpu_peak_milli = 0.0
        self._episode_live_gpu_last_milli = 0.0
        self.current_episode_number = episode_idx
        self._episodes_served += 1

        self._sync_current_pod()
        obs = self._get_obs()
        info = {
            "episode_index": episode_idx,
            "action_mask": self._current_mask.astype(np.int8),
            "strict_action_mask": self._strict_mask.astype(np.int8),
            "pod_name": self._current_pod().name,
            "obs_dim": self.obs_dim,
        }
        return obs, info

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("step() called on terminated episode. Call reset().")
        if self.sim is None:
            raise RuntimeError("simulator is not initialised")

        self._sync_current_pod()
        pod = self._current_pod()
        wait_ms = max(0, int(self.sim.current_time) - int(pod.creation_time))

        strict_mask = self._strict_mask
        no_feasible = not np.any(strict_mask)
        scheduled = False
        reason = "scheduled"
        frag_before = 0.0
        frag_after = 0.0

        if no_feasible:
            reason = "no_feasible_node"
        else:
            action_idx = int(action)
            if action_idx < 0 or action_idx >= self.node_count:
                raise ValueError(f"action index out of bounds: {action_idx}")

            if not strict_mask[action_idx]:
                if self.invalid_action_mode == "error":
                    raise ValueError(f"invalid action {action_idx} for pod {pod.name}")
                reason = "invalid_action"
            else:
                frag_before = self.sim.get_fragmentation_score(action_idx)
                scheduled = self.sim.schedule_pod(
                    pod=pod,
                    node=action_idx,
                    sub_placement_policy=self.sub_placement_policy,
                )
                if not scheduled:
                    reason = "schedule_recheck_failed"
                else:
                    frag_after = self.sim.get_fragmentation_score(action_idx)

        reward_parts = self._compute_reward(
            pod=pod,
            scheduled=scheduled,
            no_feasible=no_feasible,
            frag_before=frag_before,
            frag_after=frag_after,
            terminated=(self._pod_idx + 1 >= min(len(self._episode_pods), self.max_pods_per_episode)),
        )
        reward = reward_parts.total
        if self.reward_clip is not None:
            low, high = self.reward_clip
            reward = float(np.clip(reward, low, high))

        info = self._build_info(
            pod=pod,
            reward=reward_parts,
            scheduled=scheduled,
            reason=reason,
            action=int(action),
        )
        info["frag_before"] = frag_before
        info["frag_after"] = frag_after
        info["frag_delta"] = frag_after - frag_before
        info["frag_delta_scaled"] = (frag_after - frag_before) * self.frag_delta_scale

        self._episode_requested_gpu_milli += pod.total_gpu_milli
        if scheduled:
            self._episode_allocated_gpu_milli += pod.total_gpu_milli
            self._episode_scheduled += 1
        else:
            self._episode_failed += 1
        self._episode_frag_delta_sum += (frag_after - frag_before)
        self._episode_wait_ms_sum += float(wait_ms)
        self._episode_reward_sum += float(reward)
        live_gpu_milli = float(sum(n.used_gpu_milli for n in self.sim.nodes))
        self._episode_live_gpu_milli_sum += live_gpu_milli
        self._episode_live_gpu_peak_milli = max(self._episode_live_gpu_peak_milli, live_gpu_milli)
        self._episode_live_gpu_last_milli = live_gpu_milli

        self._pod_idx += 1
        if self._pod_idx >= min(len(self._episode_pods), self.max_pods_per_episode):
            self._terminated = True
            obs = np.zeros((self.obs_dim,), dtype=np.float32)
            terminated = True
            info["episode_metrics"] = self.get_metrics()
        else:
            self._synced_pod_idx = -1
            self._sync_current_pod()
            obs = self._get_obs()
            terminated = False

        truncated = False
        return obs, float(reward), terminated, truncated, info


def register_env() -> None:
    try:
        from gymnasium.envs.registration import register
    except Exception:
        return
    try:
        register(
            id="gpu-scheduling-v0",
            entry_point="gpu_scheduling_env:GPUSchedulingEnv",
        )
    except Exception:
        pass


def build_env_from_preprocessed_csv(
    *,
    nodes_csv: Union[str, Path],
    pods_csv: Union[str, Path],
    demand_distribution_json: Union[str, Path],
    node_count: int = 128,
    max_pods_per_episode: int = 500,
    seed: Optional[int] = None,
    **kwargs: object,
) -> GPUSchedulingEnv:
    nodes_path = Path(nodes_csv)
    pods_path = Path(pods_csv)
    demand_path = Path(demand_distribution_json)

    nodes = load_nodes(nodes_path)
    pod_rows: List[Dict[str, object]] = []
    with pods_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pod_rows.append(dict(row))
    min_pod_duration_ms = int(kwargs.pop("min_pod_duration_ms", DEFAULT_MIN_POD_DURATION_MS))
    pods = [
        _pod_from_mapping(row, i, min_pod_duration_ms=min_pod_duration_ms)
        for i, row in enumerate(pod_rows)
    ]

    gpu_only = bool(kwargs.pop("gpu_only_pods", False))
    replication_factor = int(kwargs.pop("pod_replication_factor", 1))
    if gpu_only:
        pods = [p for p in pods if p.num_gpu > 0]
    if not pods:
        raise ValueError(f"no pods available after filtering in {pods_path}")

    if replication_factor > 1:
        min_creation = min(p.creation_time for p in pods)
        max_deletion = max(p.deletion_time for p in pods)
        span = max(1, max_deletion - min_creation + 1)
        replicated: List[Pod] = []
        for rep in range(replication_factor):
            shift = rep * span
            for pod in pods:
                pod_copy = copy.deepcopy(pod)
                pod_copy.name = f"{pod.name}__rep{rep}"
                pod_copy.creation_time = int(pod.creation_time) + shift
                pod_copy.deletion_time = int(pod.deletion_time) + shift
                pod_copy.scheduled_time = None
                replicated.append(pod_copy)
        pods = replicated

    import json

    with demand_path.open("r") as f:
        payload = json.load(f)
    dist_raw = payload.get("distribution", payload)
    demand_distribution: Dict[str, Dict[Tuple[int, int], float]] = {}
    for model, model_dist in dist_raw.items():
        parsed: Dict[Tuple[int, int], float] = {}
        for key, prob in model_dist.items():
            if isinstance(key, str) and "|" in key:
                a, b = key.split("|", 1)
                parsed[(int(a), int(b))] = float(prob)
            elif isinstance(key, (tuple, list)) and len(key) == 2:
                parsed[(int(key[0]), int(key[1]))] = float(prob)
        demand_distribution[model] = parsed

    return GPUSchedulingEnv(
        nodes=nodes,
        pods=pods,
        demand_distribution=demand_distribution,
        node_count=node_count,
        max_pods_per_episode=max_pods_per_episode,
        min_pod_duration_ms=min_pod_duration_ms,
        seed=seed,
        **kwargs,
    )


register_env()
