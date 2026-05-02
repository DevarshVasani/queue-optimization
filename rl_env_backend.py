from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Q1_LACK_BOTH = "q1_lack_both"
Q2_LACK_GPU = "q2_lack_gpu"
Q3_SATISFIED = "q3_satisfied"
Q4_LACK_CPU = "q4_lack_cpu"
XL_SATISFIED = "xl_satisfied"
XR_LACK_CPU = "xr_lack_cpu"
NO_ACCESS = "no_access"

FRAG_BUCKETS = (
    Q1_LACK_BOTH,
    Q2_LACK_GPU,
    Q3_SATISFIED,
    Q4_LACK_CPU,
    XL_SATISFIED,
    XR_LACK_CPU,
    NO_ACCESS,
)


def _to_int(value: object, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def _parse_gpu_spec(raw: object) -> Tuple[str, ...]:
    if raw is None:
        return ()
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ()
    parts = [p.strip() for p in s.split("|") if p.strip()]
    return tuple(parts)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class PodResourceShape:
    cpu_milli: int
    gpu_milli: int
    num_gpu: int
    gpu_type: str


@dataclass
class PodSpec:
    name: str
    cpu_milli: int
    memory_mib: int
    num_gpu: int
    gpu_milli: int
    gpu_spec: Tuple[str, ...]
    creation_time: int
    duration: int
    deletion_time: Optional[int] = None
    qos: str = ""
    pod_phase: str = ""

    @property
    def total_gpu_milli(self) -> int:
        return self.num_gpu * self.gpu_milli

    @property
    def resource_shape(self) -> PodResourceShape:
        gpu_type = "|".join(self.gpu_spec)
        return PodResourceShape(
            cpu_milli=self.cpu_milli,
            gpu_milli=self.gpu_milli,
            num_gpu=self.num_gpu,
            gpu_type=gpu_type,
        )


@dataclass
class NodeState:
    name: str
    cpu_milli_capacity: int
    memory_mib_capacity: int
    gpu_count: int
    model: str
    cpu_milli_free: int = field(init=False)
    memory_mib_free: int = field(init=False)
    gpu_milli_free_list: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.cpu_milli_free = self.cpu_milli_capacity
        self.memory_mib_free = self.memory_mib_capacity
        self.gpu_milli_free_list = [1000 for _ in range(self.gpu_count)]

    @property
    def total_gpu_milli(self) -> int:
        return self.gpu_count * 1000

    @property
    def idle_gpu_milli(self) -> int:
        return sum(self.gpu_milli_free_list)

    @property
    def used_gpu_milli(self) -> int:
        return self.total_gpu_milli - self.idle_gpu_milli

    @property
    def used_cpu_milli(self) -> int:
        return self.cpu_milli_capacity - self.cpu_milli_free

    def copy(self) -> "NodeState":
        out = NodeState(
            name=self.name,
            cpu_milli_capacity=self.cpu_milli_capacity,
            memory_mib_capacity=self.memory_mib_capacity,
            gpu_count=self.gpu_count,
            model=self.model,
        )
        out.cpu_milli_free = self.cpu_milli_free
        out.memory_mib_free = self.memory_mib_free
        out.gpu_milli_free_list = list(self.gpu_milli_free_list)
        return out

    def can_host_cpu_mem(self, pod: PodSpec) -> bool:
        return self.cpu_milli_free >= pod.cpu_milli and self.memory_mib_free >= pod.memory_mib

    def apply_allocation(self, pod: PodSpec, gpu_indices: Sequence[int]) -> None:
        self.cpu_milli_free -= pod.cpu_milli
        self.memory_mib_free -= pod.memory_mib
        for idx in gpu_indices:
            self.gpu_milli_free_list[idx] -= pod.gpu_milli

    def release_allocation(self, pod: PodSpec, gpu_indices: Sequence[int]) -> None:
        self.cpu_milli_free += pod.cpu_milli
        self.memory_mib_free += pod.memory_mib
        for idx in gpu_indices:
            self.gpu_milli_free_list[idx] += pod.gpu_milli


@dataclass
class Allocation:
    pod: PodSpec
    node_name: str
    gpu_indices: Tuple[int, ...]
    start_time: int
    end_time: int

    @property
    def wait_time(self) -> int:
        return self.start_time - self.pod.creation_time

    @property
    def jct(self) -> int:
        return self.end_time - self.pod.creation_time


@dataclass
class TypicalPod:
    shape: PodResourceShape
    percentage: float


@dataclass
class PlacementDecision:
    node_name: str
    gpu_indices: Tuple[int, ...]


class SchedulingPolicy:
    name = "base"

    def order_pending(self, pending: Sequence[PodSpec], _: "TraceReplaySimulator") -> Sequence[PodSpec]:
        return pending

    def select_placement(self, pod: PodSpec, sim: "TraceReplaySimulator") -> Optional[PlacementDecision]:
        raise NotImplementedError


class FirstFitPolicy(SchedulingPolicy):
    name = "first_fit"

    def select_placement(self, pod: PodSpec, sim: "TraceReplaySimulator") -> Optional[PlacementDecision]:
        for node in sim.nodes:
            gpu_indices = sim.packed_gpu_indices(node, pod)
            if gpu_indices is not None:
                return PlacementDecision(node_name=node.name, gpu_indices=gpu_indices)
        return None


class RandomFitPolicy(SchedulingPolicy):
    name = "random_fit"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def select_placement(self, pod: PodSpec, sim: "TraceReplaySimulator") -> Optional[PlacementDecision]:
        candidates: List[PlacementDecision] = []
        for node in sim.nodes:
            if not sim.node_basic_access(node, pod):
                continue
            if pod.num_gpu == 0:
                candidates.append(PlacementDecision(node_name=node.name, gpu_indices=()))
                continue
            if pod.num_gpu == 1 and pod.gpu_milli < 1000:
                gpu_ids = [i for i, left in enumerate(node.gpu_milli_free_list) if left >= pod.gpu_milli]
                for idx in gpu_ids:
                    candidates.append(PlacementDecision(node_name=node.name, gpu_indices=(idx,)))
            else:
                packed = sim.packed_gpu_indices(node, pod)
                if packed is not None:
                    candidates.append(PlacementDecision(node_name=node.name, gpu_indices=packed))
        if not candidates:
            return None
        return self._rng.choice(candidates)


class BestFitPolicy(SchedulingPolicy):
    name = "best_fit"

    def select_placement(self, pod: PodSpec, sim: "TraceReplaySimulator") -> Optional[PlacementDecision]:
        best: Optional[Tuple[Tuple[int, int, str], PlacementDecision]] = None
        for node in sim.nodes:
            if not sim.node_basic_access(node, pod):
                continue
            if pod.num_gpu == 0:
                key = (node.cpu_milli_free - pod.cpu_milli, node.memory_mib_free - pod.memory_mib, node.name)
                dec = PlacementDecision(node_name=node.name, gpu_indices=())
                if best is None or key < best[0]:
                    best = (key, dec)
                continue

            if pod.num_gpu == 1 and pod.gpu_milli < 1000:
                for idx, left in enumerate(node.gpu_milli_free_list):
                    if left < pod.gpu_milli:
                        continue
                    gpu_left_after = left - pod.gpu_milli
                    key = (gpu_left_after, node.cpu_milli_free - pod.cpu_milli, node.name)
                    dec = PlacementDecision(node_name=node.name, gpu_indices=(idx,))
                    if best is None or key < best[0]:
                        best = (key, dec)
            else:
                packed = sim.packed_gpu_indices(node, pod)
                if packed is None:
                    continue
                post_idle = node.idle_gpu_milli - pod.total_gpu_milli
                key = (post_idle, node.cpu_milli_free - pod.cpu_milli, node.name)
                dec = PlacementDecision(node_name=node.name, gpu_indices=packed)
                if best is None or key < best[0]:
                    best = (key, dec)

        return best[1] if best is not None else None


class FGDPolicy(SchedulingPolicy):
    name = "fgd"

    def select_placement(self, pod: PodSpec, sim: "TraceReplaySimulator") -> Optional[PlacementDecision]:
        best_score = -1.0
        best_decision: Optional[PlacementDecision] = None

        for node in sim.nodes:
            if not sim.node_basic_access(node, pod):
                continue

            frag_before = sim.node_frag_score(node)
            if pod.num_gpu == 0:
                projected = sim.projected_node_after(node, pod, ())
                if projected is None:
                    continue
                frag_after = sim.node_frag_score(projected)
                score = _sigmoid((frag_before - frag_after) / 1000.0) * 100.0
                decision = PlacementDecision(node_name=node.name, gpu_indices=())
                if score > best_score or (
                    math.isclose(score, best_score) and best_decision is not None and node.name < best_decision.node_name
                ):
                    best_score = score
                    best_decision = decision
                continue

            if pod.num_gpu == 1 and pod.gpu_milli < 1000:
                for idx, left in enumerate(node.gpu_milli_free_list):
                    if left < pod.gpu_milli:
                        continue
                    projected = sim.projected_node_after(node, pod, (idx,))
                    if projected is None:
                        continue
                    frag_after = sim.node_frag_score(projected)
                    score = _sigmoid((frag_before - frag_after) / 1000.0) * 100.0
                    decision = PlacementDecision(node_name=node.name, gpu_indices=(idx,))
                    if score > best_score or (
                        math.isclose(score, best_score)
                        and best_decision is not None
                        and (node.name, idx) < (best_decision.node_name, best_decision.gpu_indices[0])
                    ):
                        best_score = score
                        best_decision = decision
            else:
                packed = sim.packed_gpu_indices(node, pod)
                if packed is None:
                    continue
                projected = sim.projected_node_after(node, pod, packed)
                if projected is None:
                    continue
                frag_after = sim.node_frag_score(projected)
                score = _sigmoid((frag_before - frag_after) / 1000.0) * 100.0
                decision = PlacementDecision(node_name=node.name, gpu_indices=packed)
                if score > best_score or (
                    math.isclose(score, best_score) and best_decision is not None and node.name < best_decision.node_name
                ):
                    best_score = score
                    best_decision = decision

        return best_decision


def policy_from_name(name: str, seed: int = 42) -> SchedulingPolicy:
    key = name.strip().lower()
    if key in {"first_fit", "firstfit", "fifo"}:
        return FirstFitPolicy()
    if key in {"best_fit", "bestfit"}:
        return BestFitPolicy()
    if key in {"random", "random_fit", "randomfit"}:
        return RandomFitPolicy(seed=seed)
    if key in {"fgd", "fgd_score"}:
        return FGDPolicy()
    raise ValueError(f"unknown policy: {name}")


def load_nodes_from_csv(node_csv: Path) -> List[NodeState]:
    nodes: List[NodeState] = []
    with node_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("sn", "")).strip()
            if not name:
                continue
            cpu_milli = _to_int(row.get("cpu_milli"), 0) or 0
            memory_mib = _to_int(row.get("memory_mib"), 0) or 0
            gpu = _to_int(row.get("gpu"), 0) or 0
            model = str(row.get("model", "")).strip() or "CPU"
            nodes.append(
                NodeState(
                    name=name,
                    cpu_milli_capacity=cpu_milli,
                    memory_mib_capacity=memory_mib,
                    gpu_count=gpu,
                    model=model,
                )
            )
    return nodes


def load_pods_from_csv(
    pod_csv: Path,
    *,
    include_missing_deletion: bool = False,
    fallback_duration: int = 3600,
    limit: Optional[int] = None,
) -> List[PodSpec]:
    pods: List[PodSpec] = []
    with pod_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("name", "")).strip()
            if not name:
                continue

            creation_time = _to_int(row.get("creation_time"), None)
            if creation_time is None:
                continue

            deletion_time = _to_int(row.get("deletion_time"), None)
            if deletion_time is None and not include_missing_deletion:
                continue

            if deletion_time is None:
                duration = max(1, fallback_duration)
            else:
                duration = max(1, deletion_time - creation_time)

            num_gpu = _to_int(row.get("num_gpu"), 0) or 0
            gpu_milli = _to_int(row.get("gpu_milli"), 0) or 0
            if num_gpu > 0:
                if gpu_milli <= 0:
                    gpu_milli = 1000
                gpu_milli = min(gpu_milli, 1000)
            else:
                gpu_milli = 0

            pods.append(
                PodSpec(
                    name=name,
                    cpu_milli=_to_int(row.get("cpu_milli"), 0) or 0,
                    memory_mib=_to_int(row.get("memory_mib"), 0) or 0,
                    num_gpu=num_gpu,
                    gpu_milli=gpu_milli,
                    gpu_spec=_parse_gpu_spec(row.get("gpu_spec")),
                    creation_time=creation_time,
                    duration=duration,
                    deletion_time=deletion_time,
                    qos=str(row.get("qos", "")).strip(),
                    pod_phase=str(row.get("pod_phase", "")).strip(),
                )
            )

            if limit is not None and len(pods) >= limit:
                break

    pods.sort(key=lambda p: (p.creation_time, p.name))
    return pods


def build_typical_pods(
    pods: Sequence[PodSpec],
    *,
    include_cpu_pods: bool = True,
    popularity_threshold: int = 95,
    increase_step: int = 1,
    gpu_res_weight: float = 0.0,
) -> List[TypicalPod]:
    weighted: Dict[PodResourceShape, float] = {}
    total = 0.0

    for pod in pods:
        if not include_cpu_pods and pod.num_gpu == 0:
            continue
        w = 1.0
        if gpu_res_weight > 0 and pod.gpu_milli == 1000:
            w = 1.0 + pod.num_gpu * gpu_res_weight
        shape = pod.resource_shape
        weighted[shape] = weighted.get(shape, 0.0) + w
        total += w

    if total <= 0:
        return []

    items = sorted(weighted.items(), key=lambda kv: (-kv[1], kv[0].cpu_milli, kv[0].gpu_milli, kv[0].num_gpu, kv[0].gpu_type))
    expected = max(0.0, min(100.0, float(popularity_threshold))) * total / 100.0
    step = max(1, increase_step)

    selected: List[Tuple[PodResourceShape, float]] = []
    cum = 0.0
    i = 0
    while i < len(items) and cum < expected:
        upper = min(len(items), i + step)
        while i < upper:
            selected.append(items[i])
            cum += items[i][1]
            i += 1

    if not selected:
        selected = items
        cum = total

    out: List[TypicalPod] = []
    for shape, w in selected:
        out.append(TypicalPod(shape=shape, percentage=w / cum))
    return out


class TraceReplaySimulator:
    def __init__(
        self,
        nodes: Sequence[NodeState],
        pods: Sequence[PodSpec],
        *,
        policy: SchedulingPolicy,
        typical_pods: Optional[Sequence[TypicalPod]] = None,
        frag_sample_interval: int = 1,
        max_time: Optional[int] = None,
    ) -> None:
        self.nodes: List[NodeState] = [n.copy() for n in nodes]
        self.node_by_name: Dict[str, NodeState] = {n.name: n for n in self.nodes}
        self.pods: List[PodSpec] = sorted(list(pods), key=lambda p: (p.creation_time, p.name))
        self.policy = policy
        self.typical_pods: List[TypicalPod] = list(typical_pods) if typical_pods is not None else build_typical_pods(self.pods)
        self.frag_sample_interval = max(1, frag_sample_interval)
        self.max_time = max_time

        self.total_gpu_milli = sum(n.total_gpu_milli for n in self.nodes)
        self.arrived_gpu_milli = 0
        self.current_time = self.pods[0].creation_time if self.pods else 0
        self.pending: List[PodSpec] = []
        self.running_heap: List[Tuple[int, int, Allocation]] = []
        self.running_by_pod: Dict[str, Allocation] = {}
        self.completed: List[Allocation] = []
        self.failed_to_place: List[PodSpec] = []
        self.metrics: List[Dict[str, float]] = []
        self._event_seq = 0
        self._heap_seq = 0
        self._completion_epoch = 0
        self._pending_attempt_epoch: Dict[str, int] = {}
        self._frag_score_cache: Dict[Tuple[object, ...], float] = {}
        self._frag_amount_cache: Dict[Tuple[object, ...], Dict[str, float]] = {}
        self._frag_cache_max = 250000
        self._last_frag = self.cluster_frag_metrics()

    def is_node_accessible_to_pod(self, node: NodeState, pod: PodSpec) -> bool:
        if pod.num_gpu == 0:
            return True
        if not pod.gpu_spec:
            return True
        return node.model in pod.gpu_spec

    def node_basic_access(self, node: NodeState, pod: PodSpec) -> bool:
        if not node.can_host_cpu_mem(pod):
            return False
        if pod.num_gpu == 0:
            return True
        if node.gpu_count < pod.num_gpu:
            return False
        if not self.is_node_accessible_to_pod(node, pod):
            return False
        eligible = sum(1 for left in node.gpu_milli_free_list if left >= pod.gpu_milli)
        return eligible >= pod.num_gpu

    def packed_gpu_indices(self, node: NodeState, pod: PodSpec) -> Optional[Tuple[int, ...]]:
        if not self.node_basic_access(node, pod):
            return None
        if pod.num_gpu == 0:
            return ()
        pairs = sorted(enumerate(node.gpu_milli_free_list), key=lambda x: (x[1], x[0]))
        chosen: List[int] = []
        for idx, left in pairs:
            if left >= pod.gpu_milli:
                chosen.append(idx)
                if len(chosen) == pod.num_gpu:
                    break
        if len(chosen) != pod.num_gpu:
            return None
        return tuple(chosen)

    def projected_node_after(self, node: NodeState, pod: PodSpec, gpu_indices: Sequence[int]) -> Optional[NodeState]:
        if not node.can_host_cpu_mem(pod):
            return None
        clone = node.copy()
        if len(gpu_indices) != pod.num_gpu:
            return None
        if len(set(gpu_indices)) != len(gpu_indices):
            return None
        for idx in gpu_indices:
            if idx < 0 or idx >= clone.gpu_count:
                return None
            if clone.gpu_milli_free_list[idx] < pod.gpu_milli:
                return None
        clone.apply_allocation(pod, gpu_indices)
        return clone

    def can_node_host_pod_on_gpu_memory(self, node: NodeState, shape: PodResourceShape) -> bool:
        if shape.gpu_milli <= 0 or shape.num_gpu <= 0:
            return True
        req = shape.num_gpu
        for left in node.gpu_milli_free_list:
            if left >= shape.gpu_milli:
                req -= 1
                if req <= 0:
                    return True
        return False

    def node_pod_frag_type(self, node: NodeState, shape: PodResourceShape) -> str:
        if shape.gpu_milli == 0 or shape.num_gpu == 0:
            return XL_SATISFIED if node.cpu_milli_free >= shape.cpu_milli else XR_LACK_CPU

        if shape.gpu_type:
            types = tuple(t for t in shape.gpu_type.split("|") if t)
            if types and node.model not in types:
                return NO_ACCESS

        if self.can_node_host_pod_on_gpu_memory(node, shape):
            return Q3_SATISFIED if node.cpu_milli_free >= shape.cpu_milli else Q4_LACK_CPU

        return Q2_LACK_GPU if node.cpu_milli_free >= shape.cpu_milli else Q1_LACK_BOTH

    def node_frag_amount(self, node: NodeState) -> Dict[str, float]:
        key = self._node_state_key(node)
        cached = self._frag_amount_cache.get(key)
        if cached is not None:
            return cached

        out = {k: 0.0 for k in FRAG_BUCKETS}
        total_left = float(node.idle_gpu_milli)
        if total_left <= 0 or not self.typical_pods:
            self._frag_amount_cache[key] = out
            return out

        for typical in self.typical_pods:
            shape = typical.shape
            freq = typical.percentage
            frag_type = self.node_pod_frag_type(node, shape)
            if frag_type == Q3_SATISFIED:
                gpu_frag = sum(left for left in node.gpu_milli_free_list if left < shape.gpu_milli)
                out[Q2_LACK_GPU] += freq * float(gpu_frag)
                out[Q3_SATISFIED] += freq * float(max(0, node.idle_gpu_milli - gpu_frag))
            else:
                out[frag_type] += freq * total_left

        self._frag_amount_cache[key] = out
        self._maybe_reset_frag_caches()
        return out

    def node_frag_score(self, node: NodeState) -> float:
        key = self._node_state_key(node)
        cached = self._frag_score_cache.get(key)
        if cached is not None:
            return cached

        frag = self.node_frag_amount(node)
        score = sum(v for k, v in frag.items() if k != Q3_SATISFIED)
        self._frag_score_cache[key] = score
        self._maybe_reset_frag_caches()
        return score

    def _node_state_key(self, node: NodeState) -> Tuple[object, ...]:
        return (node.model, node.cpu_milli_free, node.memory_mib_free, tuple(node.gpu_milli_free_list))

    def _maybe_reset_frag_caches(self) -> None:
        if len(self._frag_score_cache) > self._frag_cache_max:
            self._frag_score_cache.clear()
        if len(self._frag_amount_cache) > self._frag_cache_max:
            self._frag_amount_cache.clear()

    def cluster_frag_metrics(self) -> Dict[str, float]:
        buckets = {k: 0.0 for k in FRAG_BUCKETS}
        for node in self.nodes:
            node_frag = self.node_frag_amount(node)
            for k in FRAG_BUCKETS:
                buckets[k] += node_frag[k]

        idle_gpu_milli = sum(buckets.values())
        frag_gpu_milli = sum(v for k, v in buckets.items() if k != Q3_SATISFIED)
        q124 = buckets[Q1_LACK_BOTH] + buckets[Q2_LACK_GPU] + buckets[Q4_LACK_CPU]

        frag_ratio = 0.0 if idle_gpu_milli <= 0 else 100.0 * frag_gpu_milli / idle_gpu_milli
        q124_ratio = 0.0 if idle_gpu_milli <= 0 else 100.0 * q124 / idle_gpu_milli

        out = {
            "origin_milli": frag_gpu_milli,
            "origin_ratio": frag_ratio,
            "origin_q124": q124_ratio,
        }
        out.update(buckets)
        return out

    def _allocate(self, pod: PodSpec, decision: PlacementDecision, now: int) -> None:
        node = self.node_by_name[decision.node_name]
        projected = self.projected_node_after(node, pod, decision.gpu_indices)
        if projected is None:
            return
        node.apply_allocation(pod, decision.gpu_indices)

        alloc = Allocation(
            pod=pod,
            node_name=node.name,
            gpu_indices=tuple(decision.gpu_indices),
            start_time=now,
            end_time=now + pod.duration,
        )
        self.running_by_pod[pod.name] = alloc
        heapq.heappush(self.running_heap, (alloc.end_time, self._heap_seq, alloc))
        self._heap_seq += 1
        self._pending_attempt_epoch.pop(pod.name, None)

    def _release_completed(self, now: int) -> None:
        completed_now = 0
        while self.running_heap and self.running_heap[0][0] <= now:
            _, _, alloc = heapq.heappop(self.running_heap)
            if alloc.pod.name not in self.running_by_pod:
                continue
            del self.running_by_pod[alloc.pod.name]
            node = self.node_by_name[alloc.node_name]
            node.release_allocation(alloc.pod, alloc.gpu_indices)
            self.completed.append(alloc)
            completed_now += 1
        if completed_now > 0:
            self._completion_epoch += 1

    def _schedule_pending(self, now: int) -> bool:
        any_progress = False
        ordered = list(self.policy.order_pending(list(self.pending), self))
        pending_set = {p.name for p in self.pending}

        for pod in ordered:
            if pod.name not in pending_set:
                continue
            if self._pending_attempt_epoch.get(pod.name) == self._completion_epoch:
                continue
            self._pending_attempt_epoch[pod.name] = self._completion_epoch
            decision = self.policy.select_placement(pod, self)
            if decision is None:
                continue
            self._allocate(pod, decision, now)
            self.pending = [p for p in self.pending if p.name != pod.name]
            pending_set.discard(pod.name)
            any_progress = True

        return any_progress

    def _record_metrics(self, now: int) -> None:
        used_gpu_milli = sum(n.used_gpu_milli for n in self.nodes)
        idle_gpu_milli = self.total_gpu_milli - used_gpu_milli

        used_nodes = 0
        used_gpus = 0
        for n in self.nodes:
            if n.gpu_count == 0:
                if n.used_cpu_milli > 0:
                    used_nodes += 1
                continue
            full_free = sum(1 for x in n.gpu_milli_free_list if x == 1000)
            if full_free < n.gpu_count or n.used_cpu_milli > 0:
                used_nodes += 1
                used_gpus += n.gpu_count

        alloc_ratio = 0.0 if self.total_gpu_milli == 0 else 100.0 * used_gpu_milli / self.total_gpu_milli
        arrival_ratio = 0.0 if self.total_gpu_milli == 0 else 100.0 * self.arrived_gpu_milli / self.total_gpu_milli
        idle_fraction = 0.0 if self.total_gpu_milli == 0 else idle_gpu_milli / self.total_gpu_milli
        unmet_gpu_milli = max(0, self.arrived_gpu_milli - used_gpu_milli)
        unallocated_fraction = 0.0 if self.arrived_gpu_milli <= 0 else unmet_gpu_milli / self.arrived_gpu_milli

        if self._event_seq % self.frag_sample_interval == 0:
            self._last_frag = self.cluster_frag_metrics()

        row = {
            "event": self._event_seq,
            "time": now,
            "used_nodes": used_nodes,
            "used_gpus": used_gpus,
            "used_gpu_milli": used_gpu_milli,
            "total_gpus": self.total_gpu_milli // 1000,
            "arrived_gpu_milli": self.arrived_gpu_milli,
            "allocation_ratio": alloc_ratio,
            "arrival_ratio": arrival_ratio,
            "idle_gpu_fraction": idle_fraction,
            "unallocated_gpu_fraction": unallocated_fraction,
        }
        row.update(self._last_frag)
        self.metrics.append(row)

    def run(self) -> Dict[str, object]:
        pod_idx = 0
        n_pods = len(self.pods)

        if n_pods == 0:
            return {
                "policy": self.policy.name,
                "total_pods": 0,
                "completed_pods": 0,
                "pending_pods": 0,
                "metrics": [],
            }

        while True:
            next_arrival = self.pods[pod_idx].creation_time if pod_idx < n_pods else None
            next_completion = self.running_heap[0][0] if self.running_heap else None

            if next_arrival is None and next_completion is None:
                if self.pending:
                    progressed = self._schedule_pending(self.current_time)
                    self._record_metrics(self.current_time)
                    self._event_seq += 1
                    if not progressed:
                        self.failed_to_place.extend(self.pending)
                        self.pending.clear()
                break

            candidates = [t for t in (next_arrival, next_completion) if t is not None]
            now = min(candidates)
            if self.max_time is not None and now > self.max_time:
                break
            self.current_time = now

            self._release_completed(now)

            while pod_idx < n_pods and self.pods[pod_idx].creation_time <= now:
                pod = self.pods[pod_idx]
                self.pending.append(pod)
                self._pending_attempt_epoch[pod.name] = -1
                self.arrived_gpu_milli += pod.total_gpu_milli
                pod_idx += 1

            self._schedule_pending(now)
            self._record_metrics(now)
            self._event_seq += 1

        wait_times = [a.wait_time for a in self.completed]
        jcts = [a.jct for a in self.completed]
        makespan = max((a.end_time for a in self.completed), default=self.current_time)

        summary = {
            "policy": self.policy.name,
            "total_pods": len(self.pods),
            "completed_pods": len(self.completed),
            "pending_pods": len(self.pending),
            "unscheduled_pods": len(self.failed_to_place),
            "avg_wait_time": float(mean(wait_times)) if wait_times else 0.0,
            "avg_jct": float(mean(jcts)) if jcts else 0.0,
            "makespan": makespan,
            "metrics": self.metrics,
            "unscheduled_pod_names": [p.name for p in self.failed_to_place],
        }
        return summary


def run_trace_replay(
    *,
    node_csv: Path,
    pod_csv: Path,
    policy_name: str = "fgd",
    seed: int = 42,
    include_missing_deletion: bool = False,
    fallback_duration: int = 3600,
    pod_limit: Optional[int] = None,
    frag_sample_interval: int = 1,
) -> Dict[str, object]:
    nodes = load_nodes_from_csv(node_csv)
    pods = load_pods_from_csv(
        pod_csv,
        include_missing_deletion=include_missing_deletion,
        fallback_duration=fallback_duration,
        limit=pod_limit,
    )
    policy = policy_from_name(policy_name, seed=seed)
    typical = build_typical_pods(pods)
    sim = TraceReplaySimulator(
        nodes=nodes,
        pods=pods,
        policy=policy,
        typical_pods=typical,
        frag_sample_interval=frag_sample_interval,
    )
    return sim.run()


def _default_paths() -> Tuple[Path, Path]:
    base = Path(__file__).resolve().parent / "clusterdata" / "cluster-trace-gpu-v2023" / "csv"
    return base / "openb_node_list_gpu_node.csv", base / "openb_pod_list_default.csv"


def main() -> None:
    default_node, default_pod = _default_paths()
    parser = argparse.ArgumentParser(description="Replay GPU cluster trace with pluggable scheduling policy")
    parser.add_argument("--node-csv", type=Path, default=default_node)
    parser.add_argument("--pod-csv", type=Path, default=default_pod)
    parser.add_argument("--policy", type=str, default="fgd", help="fgd|best_fit|first_fit|random_fit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pod-limit", type=int, default=None)
    parser.add_argument("--include-missing-deletion", action="store_true")
    parser.add_argument("--fallback-duration", type=int, default=3600)
    parser.add_argument("--frag-sample-interval", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()

    summary = run_trace_replay(
        node_csv=args.node_csv,
        pod_csv=args.pod_csv,
        policy_name=args.policy,
        seed=args.seed,
        include_missing_deletion=args.include_missing_deletion,
        fallback_duration=args.fallback_duration,
        pod_limit=args.pod_limit,
        frag_sample_interval=args.frag_sample_interval,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary to {args.output}")
    else:
        final = summary["metrics"][-1] if summary["metrics"] else {}
        print(json.dumps(
            {
                "policy": summary["policy"],
                "total_pods": summary["total_pods"],
                "completed_pods": summary["completed_pods"],
                "unscheduled_pods": summary["unscheduled_pods"],
                "avg_wait_time": summary["avg_wait_time"],
                "avg_jct": summary["avg_jct"],
                "makespan": summary["makespan"],
                "final_metrics": final,
            },
            indent=2,
        ))


if __name__ == "__main__":
    main()
