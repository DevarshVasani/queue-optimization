from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import random
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


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
    return tuple(x.strip() for x in s.split("|") if x.strip())


def qos_priority(qos: str) -> int:
    qos = qos.strip().lower()
    if qos == "ls":
        return 3
    if qos == "burstable":
        return 2
    if qos in {"be", "besteffort", "best-effort", "best_effort"}:
        return 1
    return 0


@dataclass
class GPU:
    allocated_milli: int = 0
    gpu_type: str = ""

    @property
    def free_milli(self) -> int:
        return 1000 - self.allocated_milli


@dataclass
class Node:
    node_id: str
    cpu_total: int
    memory_total: int
    model: str
    gpu_list: List[GPU] = field(default_factory=list)
    cpu_avail: int = field(init=False)
    memory_avail: int = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cpu_avail = self.cpu_total
        self.memory_avail = self.memory_total
        for gpu in self.gpu_list:
            gpu.allocated_milli = 0

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_list)

    @property
    def used_gpu_milli(self) -> int:
        return sum(g.allocated_milli for g in self.gpu_list)

    @property
    def free_gpu_milli(self) -> int:
        return sum(g.free_milli for g in self.gpu_list)

    @property
    def partially_used_gpu_count(self) -> int:
        return sum(1 for g in self.gpu_list if 0 < g.allocated_milli < 1000)

    @property
    def used_cpu(self) -> int:
        return self.cpu_total - self.cpu_avail


@dataclass
class Pod:
    name: str
    cpu_milli: int
    memory_mib: int
    num_gpu: int
    gpu_milli: int
    gpu_spec: Tuple[str, ...]
    qos: str
    priority: int
    creation_time: int
    deletion_time: int
    scheduled_time: Optional[int] = None
    assigned_node: Optional[str] = None
    assigned_gpus: Tuple[int, ...] = ()

    @property
    def total_gpu_milli(self) -> int:
        return self.num_gpu * self.gpu_milli


@dataclass
class RunningPod:
    pod_name: str
    node_id: str
    gpu_indices: Tuple[int, ...]
    cpu_milli: int
    memory_mib: int
    gpu_milli: int
    deletion_time: int
    creation_time: int
    scheduled_time: int


@dataclass
class FeasibleNode:
    node_id: str
    gpu_candidates: List[Tuple[int, ...]]
    node_ref: Node


@dataclass
class DecisionContext:
    current_time: int
    pod: Pod
    feasible_nodes: List[FeasibleNode]


@dataclass
class StepResult:
    pod_name: str
    current_time: int
    scheduled: bool
    failed: bool
    reason: str
    chosen_node: Optional[str]
    chosen_gpus: Tuple[int, ...]
    feasible_node_ids: List[str]
    metrics: Dict[str, float]
    done: bool


def load_nodes(node_csv: Path) -> List[Node]:
    nodes: List[Node] = []
    with node_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = str(row.get("sn", "")).strip()
            if not node_id:
                continue
            cpu_total = _to_int(row.get("cpu_milli"), 0) or 0
            memory_total = _to_int(row.get("memory_mib"), 0) or 0
            gpu_count = _to_int(row.get("gpu"), 0) or 0
            model = str(row.get("model", "")).strip() or "CPU"
            gpus = [GPU(allocated_milli=0, gpu_type=model) for _ in range(gpu_count)]
            nodes.append(
                Node(
                    node_id=node_id,
                    cpu_total=cpu_total,
                    memory_total=memory_total,
                    model=model,
                    gpu_list=gpus,
                )
            )
    return nodes


def load_pods(
    pod_csv: Path,
    *,
    drop_missing_deletion: bool = False,
    default_duration: int = 3600,
    min_pod_duration: int = 0,
    limit: Optional[int] = None,
) -> List[Pod]:
    pods: List[Pod] = []
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
            if deletion_time is None:
                if drop_missing_deletion:
                    continue
                deletion_time = creation_time + max(1, default_duration)

            min_duration = max(0, int(min_pod_duration))
            min_valid_deletion = creation_time + max(1, min_duration)
            if deletion_time < min_valid_deletion:
                deletion_time = min_valid_deletion

            num_gpu = _to_int(row.get("num_gpu"), 0) or 0
            gpu_milli = _to_int(row.get("gpu_milli"), 0) or 0
            if num_gpu > 0:
                gpu_milli = 1000 if gpu_milli <= 0 else min(1000, gpu_milli)
            else:
                gpu_milli = 0

            qos = str(row.get("qos", "")).strip()
            explicit_priority = _to_int(row.get("priority"), None)
            scheduled_time = _to_int(row.get("scheduled_time"), None)
            pod = Pod(
                name=name,
                cpu_milli=_to_int(row.get("cpu_milli"), 0) or 0,
                memory_mib=_to_int(row.get("memory_mib"), 0) or 0,
                num_gpu=num_gpu,
                gpu_milli=gpu_milli,
                gpu_spec=_parse_gpu_spec(row.get("gpu_spec")),
                qos=qos,
                priority=qos_priority(qos) if explicit_priority is None else explicit_priority,
                creation_time=creation_time,
                deletion_time=deletion_time,
                scheduled_time=scheduled_time,
            )
            pods.append(pod)
            if limit is not None and len(pods) >= limit:
                break

    pods.sort(key=lambda p: (p.creation_time, -p.priority, p.name))
    return pods


def build_gpu_demand_distribution(
    pods: Sequence[Pod],
    models: Sequence[str],
) -> Dict[str, Dict[Tuple[int, int], float]]:
    model_set = sorted(set(models))
    counter: Dict[str, Dict[Tuple[int, int], int]] = {m: {} for m in model_set}

    for pod in pods:
        if pod.num_gpu <= 0:
            continue
        demand = (pod.num_gpu, pod.gpu_milli)
        if pod.gpu_spec:
            targets = [m for m in model_set if m in pod.gpu_spec]
        else:
            targets = model_set
        if not targets:
            continue
        for model in targets:
            counts = counter[model]
            counts[demand] = counts.get(demand, 0) + 1

    distribution: Dict[str, Dict[Tuple[int, int], float]] = {}
    for model in model_set:
        counts = counter[model]
        total = sum(counts.values())
        if total <= 0:
            distribution[model] = {}
            continue
        distribution[model] = {d: c / total for d, c in counts.items()}
    return distribution


class SchedulerPolicy:
    name = "base"

    def choose_action(self, context: DecisionContext, sim: "ClusterSimulator") -> Optional[Dict[str, object]]:
        raise NotImplementedError


class RandomPolicy(SchedulerPolicy):
    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, context: DecisionContext, sim: "ClusterSimulator") -> Optional[Dict[str, object]]:
        if not context.feasible_nodes:
            return None
        item = self.rng.choice(context.feasible_nodes)
        return {
            "node_id": item.node_id,
            "gpu_indices": item.gpu_candidates[0] if item.gpu_candidates else (),
        }


class BestFitPolicy(SchedulerPolicy):
    name = "best_fit"

    def choose_action(self, context: DecisionContext, sim: "ClusterSimulator") -> Optional[Dict[str, object]]:
        best: Optional[Tuple[Tuple[int, int, str], Dict[str, object]]] = None
        pod = context.pod

        for item in context.feasible_nodes:
            for gpus in item.gpu_candidates[:1]:
                remain = item.node_ref.free_gpu_milli - pod.total_gpu_milli
                cpu_remain = item.node_ref.cpu_avail - pod.cpu_milli
                key = (remain, cpu_remain, item.node_id)
                action = {"node_id": item.node_id, "gpu_indices": gpus}
                if best is None or key < best[0]:
                    best = (key, action)

        return None if best is None else best[1]


class DotProductPolicy(SchedulerPolicy):
    name = "dot_product"

    def choose_action(self, context: DecisionContext, sim: "ClusterSimulator") -> Optional[Dict[str, object]]:
        best_score = None
        best_action: Optional[Dict[str, object]] = None
        pod = context.pod
        pod_vec = (
            pod.cpu_milli,
            pod.memory_mib,
            pod.total_gpu_milli,
            pod.gpu_milli,
        )

        for item in context.feasible_nodes:
            node = item.node_ref
            node_vec = (
                node.cpu_avail,
                node.memory_avail,
                node.free_gpu_milli,
                max((g.free_milli for g in node.gpu_list), default=0),
            )
            score = sum(a * b for a, b in zip(pod_vec, node_vec))
            if best_score is None or score > best_score:
                best_score = score
                best_action = {
                    "node_id": item.node_id,
                    "gpu_indices": item.gpu_candidates[0] if item.gpu_candidates else (),
                }

        return best_action


class GPUPackingPolicy(SchedulerPolicy):
    name = "gpu_packing"

    def choose_action(self, context: DecisionContext, sim: "ClusterSimulator") -> Optional[Dict[str, object]]:
        best: Optional[Tuple[Tuple[int, int, str], Dict[str, object]]] = None
        pod = context.pod

        for item in context.feasible_nodes:
            node = item.node_ref
            for gpus in item.gpu_candidates:
                after = sim.simulate_node_after(item.node_ref, pod, gpus)
                if after is None:
                    continue
                partial = after.partially_used_gpu_count
                free = after.free_gpu_milli
                key = (-partial, free, item.node_id)
                action = {"node_id": item.node_id, "gpu_indices": gpus}
                if best is None or key < best[0]:
                    best = (key, action)

        return None if best is None else best[1]


class GPUClusteringPolicy(SchedulerPolicy):
    name = "gpu_clustering"

    def choose_action(self, context: DecisionContext, sim: "ClusterSimulator") -> Optional[Dict[str, object]]:
        if not context.feasible_nodes:
            return None

        pod = context.pod
        gpu_required = pod.num_gpu > 0
        preferred: List[FeasibleNode] = []
        fallback: List[FeasibleNode] = []

        for item in context.feasible_nodes:
            has_gpu = item.node_ref.gpu_count > 0
            if gpu_required and has_gpu:
                preferred.append(item)
            elif (not gpu_required) and (not has_gpu):
                preferred.append(item)
            else:
                fallback.append(item)

        target = preferred if preferred else fallback
        if not target:
            return None

        item = sorted(target, key=lambda x: x.node_id)[0]
        return {
            "node_id": item.node_id,
            "gpu_indices": item.gpu_candidates[0] if item.gpu_candidates else (),
        }


class FGDPolicy(SchedulerPolicy):
    name = "fgd"

    def choose_action(self, context: DecisionContext, sim: "ClusterSimulator") -> Optional[Dict[str, object]]:
        best: Optional[Tuple[Tuple[float, int, str], Dict[str, object]]] = None
        pod = context.pod

        for item in context.feasible_nodes:
            node_before = item.node_ref
            f_before = sim.fragmentation_score(node_before)

            for gpus in item.gpu_candidates:
                node_after = sim.simulate_node_after(node_before, pod, gpus)
                if node_after is None:
                    continue
                f_after = sim.fragmentation_score(node_after)
                delta = f_after - f_before
                cpu_remaining = node_after.cpu_avail
                key = (delta, -cpu_remaining, item.node_id)
                action = {"node_id": item.node_id, "gpu_indices": gpus}
                if best is None or key < best[0]:
                    best = (key, action)

        return None if best is None else best[1]


def policy_from_name(name: str, seed: int = 42) -> SchedulerPolicy:
    key = name.strip().lower()
    if key in {"random", "random_fit"}:
        return RandomPolicy(seed=seed)
    if key in {"best_fit", "bestfit"}:
        return BestFitPolicy()
    if key in {"dot", "dot_product", "dotproduct"}:
        return DotProductPolicy()
    if key in {"gpu_packing", "packing"}:
        return GPUPackingPolicy()
    if key in {"gpu_clustering", "clustering"}:
        return GPUClusteringPolicy()
    if key in {"fgd", "fgd_score"}:
        return FGDPolicy()
    raise ValueError(f"unknown policy: {name}")


class ClusterSimulator:
    def __init__(
        self,
        nodes: Sequence[Node],
        pods: Sequence[Pod],
        *,
        demand_distribution: Optional[Dict[str, Dict[Tuple[int, int], float]]] = None,
        fragmentation_mode: str = "fgd",
        fragmentation_scale: float = 1.0,
        record_history: bool = False,
    ) -> None:
        self.nodes_template = copy.deepcopy(list(nodes))
        self.pods_template = copy.deepcopy(list(pods))
        models = [n.model for n in self.nodes_template if n.gpu_count > 0]
        self.demand_distribution = (
            demand_distribution
            if demand_distribution is not None
            else build_gpu_demand_distribution(self.pods_template, models)
        )
        mode = fragmentation_mode.strip().lower()
        if mode not in {"fgd", "utilization"}:
            raise ValueError("fragmentation_mode must be one of: fgd, utilization")
        self.fragmentation_mode = mode
        self.fragmentation_scale = float(fragmentation_scale)
        self.record_history = record_history
        self.reset()

    def reset(self) -> DecisionContext:
        self.nodes: List[Node] = copy.deepcopy(self.nodes_template)
        self.node_map: Dict[str, Node] = {n.node_id: n for n in self.nodes}
        self.pods: List[Pod] = copy.deepcopy(self.pods_template)
        self.pod_map: Dict[str, Pod] = {p.name: p for p in self.pods}

        self.current_time = self.pods[0].creation_time if self.pods else 0
        self.pending_pods: List[str] = []
        self.running_heap: List[Tuple[int, int, str]] = []
        self.running_map: Dict[str, RunningPod] = {}
        self.completed: List[RunningPod] = []
        self.failed: List[str] = []

        self.incoming_index = 0
        self.done = len(self.pods) == 0
        self._running_seq = 0
        self._prepared: Optional[DecisionContext] = None
        self._frag_cache: Dict[Tuple[str, Tuple[int, ...]], float] = {}
        self._placement_cache: Dict[Tuple[str, str], List[Tuple[int, ...]]] = {}

        self.total_gpu_capacity_milli = sum(n.gpu_count * 1000 for n in self.nodes)
        self.arrived_gpu_milli_sum = 0
        self.allocated_gpu_milli_sum = 0
        self.event_metrics: List[Dict[str, float]] = []

        if self.done:
            raise ValueError("no pods loaded")

        return self.prepare_next_decision()

    def clone(self) -> "ClusterSimulator":
        return copy.deepcopy(self)

    @property
    def node_states(self) -> List[Dict[str, object]]:
        states: List[Dict[str, object]] = []
        for node in self.nodes:
            states.append(
                {
                    "node_id": node.node_id,
                    "cpu_total": node.cpu_total,
                    "cpu_avail": node.cpu_avail,
                    "memory_total": node.memory_total,
                    "memory_avail": node.memory_avail,
                    "gpu_count": node.gpu_count,
                    "model": node.model,
                    "gpu_allocated_milli": [g.allocated_milli for g in node.gpu_list],
                    "gpu_free_milli": [g.free_milli for g in node.gpu_list],
                }
            )
        return states

    def _resolve_node(self, node: Union[int, str, Node]) -> Node:
        if isinstance(node, Node):
            return node
        if isinstance(node, int):
            if node < 0 or node >= len(self.nodes):
                raise IndexError(f"node index out of range: {node}")
            return self.nodes[node]
        node_id = str(node)
        if node_id not in self.node_map:
            raise KeyError(f"unknown node id: {node_id}")
        return self.node_map[node_id]

    def advance_to_time(self, timestamp: int) -> List[Dict[str, object]]:
        self.current_time = max(self.current_time, int(timestamp))
        self._process_completions()
        self._prepared = None
        return self.node_states

    def get_fragmentation_score(self, node: Union[int, str, Node]) -> float:
        node_ref = self._resolve_node(node)
        return float(self.fragmentation_score(node_ref))

    def check_feasibility(self, pod: Pod, node: Union[int, str, Node]) -> bool:
        node_ref = self._resolve_node(node)
        if node_ref.cpu_avail < pod.cpu_milli:
            return False
        if node_ref.memory_avail < pod.memory_mib:
            return False
        if not self._can_match_model(node_ref, pod):
            return False
        return len(self._candidate_gpu_sets(node_ref, pod)) > 0

    def _select_gpu_indices_for_policy(
        self,
        node: Node,
        pod: Pod,
        candidates: Sequence[Tuple[int, ...]],
        policy: str,
    ) -> Tuple[int, ...]:
        if not candidates:
            return ()
        key = policy.strip().lower()
        if key in {"first_fit", "first", "default"}:
            return tuple(candidates[0])
        if key in {"most_used_first", "pack", "packing"}:
            scored = sorted(
                candidates,
                key=lambda idxs: (
                    -sum(node.gpu_list[i].allocated_milli for i in idxs),
                    sum(node.gpu_list[i].free_milli - pod.gpu_milli for i in idxs),
                    idxs,
                ),
            )
            return tuple(scored[0])
        raise ValueError(f"unknown sub_placement_policy: {policy}")

    def schedule_pod(
        self,
        pod: Pod,
        node: Union[int, str, Node],
        sub_placement_policy: str = "most_used_first",
    ) -> bool:
        node_ref = self._resolve_node(node)
        if not self.check_feasibility(pod, node_ref):
            return False

        candidates = self._candidate_gpu_sets(node_ref, pod)
        gpu_indices = self._select_gpu_indices_for_policy(
            node_ref,
            pod,
            candidates,
            policy=sub_placement_policy,
        )
        if len(gpu_indices) != pod.num_gpu:
            return False

        node_ref.cpu_avail -= pod.cpu_milli
        node_ref.memory_avail -= pod.memory_mib
        for idx in gpu_indices:
            node_ref.gpu_list[idx].allocated_milli += pod.gpu_milli

        pod.scheduled_time = self.current_time
        pod.assigned_node = node_ref.node_id
        pod.assigned_gpus = tuple(gpu_indices)

        running = RunningPod(
            pod_name=pod.name,
            node_id=node_ref.node_id,
            gpu_indices=tuple(gpu_indices),
            cpu_milli=pod.cpu_milli,
            memory_mib=pod.memory_mib,
            gpu_milli=pod.gpu_milli,
            deletion_time=pod.deletion_time,
            creation_time=pod.creation_time,
            scheduled_time=self.current_time,
        )
        self.running_map[pod.name] = running
        heapq.heappush(self.running_heap, (running.deletion_time, self._running_seq, pod.name))
        self._running_seq += 1
        self.allocated_gpu_milli_sum += pod.total_gpu_milli
        self._prepared = None
        return True

    def get_state(self) -> Dict[str, object]:
        return {
            "current_time": self.current_time,
            "incoming_index": self.incoming_index,
            "pending_pods": list(self.pending_pods),
            "running_pods": [
                {
                    "pod_name": r.pod_name,
                    "node_id": r.node_id,
                    "gpu_indices": list(r.gpu_indices),
                    "deletion_time": r.deletion_time,
                    "scheduled_time": r.scheduled_time,
                }
                for r in self.running_map.values()
            ],
            "nodes": [
                {
                    "node_id": n.node_id,
                    "cpu_avail": n.cpu_avail,
                    "memory_avail": n.memory_avail,
                    "gpu_allocated": [g.allocated_milli for g in n.gpu_list],
                }
                for n in self.nodes
            ],
            "arrived_gpu_milli_sum": self.arrived_gpu_milli_sum,
            "allocated_gpu_milli_sum": self.allocated_gpu_milli_sum,
            "failed": list(self.failed),
            "done": self.done,
        }

    def set_state(self, state: Dict[str, object]) -> None:
        self.current_time = int(state["current_time"])
        self.incoming_index = int(state["incoming_index"])
        self.pending_pods = list(state["pending_pods"])
        self.arrived_gpu_milli_sum = int(state["arrived_gpu_milli_sum"])
        self.allocated_gpu_milli_sum = int(state["allocated_gpu_milli_sum"])
        self.failed = list(state["failed"])
        self.done = bool(state["done"])

        node_state = {x["node_id"]: x for x in state["nodes"]}  # type: ignore[index]
        for node in self.nodes:
            ns = node_state[node.node_id]
            node.cpu_avail = int(ns["cpu_avail"])
            node.memory_avail = int(ns["memory_avail"])
            for i, milli in enumerate(ns["gpu_allocated"]):
                node.gpu_list[i].allocated_milli = int(milli)

        self.running_map = {}
        self.running_heap = []
        self._running_seq = 0
        for item in state["running_pods"]:  # type: ignore[index]
            pod_name = str(item["pod_name"])
            pod = self.pod_map[pod_name]
            running = RunningPod(
                pod_name=pod_name,
                node_id=str(item["node_id"]),
                gpu_indices=tuple(int(x) for x in item["gpu_indices"]),
                cpu_milli=pod.cpu_milli,
                memory_mib=pod.memory_mib,
                gpu_milli=pod.gpu_milli,
                deletion_time=int(item["deletion_time"]),
                creation_time=pod.creation_time,
                scheduled_time=int(item["scheduled_time"]),
            )
            self.running_map[pod_name] = running
            heapq.heappush(self.running_heap, (running.deletion_time, self._running_seq, pod_name))
            self._running_seq += 1
        self._prepared = None

    def _process_completions(self) -> None:
        while self.running_heap and self.running_heap[0][0] <= self.current_time:
            _, _, pod_name = heapq.heappop(self.running_heap)
            running = self.running_map.pop(pod_name, None)
            if running is None:
                continue
            node = self.node_map[running.node_id]
            node.cpu_avail += running.cpu_milli
            node.memory_avail += running.memory_mib
            for idx in running.gpu_indices:
                node.gpu_list[idx].allocated_milli -= running.gpu_milli
            self.completed.append(running)

    def _can_match_model(self, node: Node, pod: Pod) -> bool:
        if pod.num_gpu == 0:
            return True
        if not pod.gpu_spec:
            return True
        return node.model in pod.gpu_spec

    def _candidate_gpu_sets(self, node: Node, pod: Pod) -> List[Tuple[int, ...]]:
        cache_key = (node.node_id, pod.name)
        cached = self._placement_cache.get(cache_key)
        if cached is not None:
            return cached

        if pod.num_gpu == 0:
            out = [()]
            self._placement_cache[cache_key] = out
            return out
        eligible = [i for i, g in enumerate(node.gpu_list) if g.free_milli >= pod.gpu_milli]
        if len(eligible) < pod.num_gpu:
            self._placement_cache[cache_key] = []
            return []
        if pod.num_gpu == 1:
            out = [(i,) for i in eligible]
            self._placement_cache[cache_key] = out
            return out

        out = [tuple(c) for c in combinations(eligible, pod.num_gpu)]
        out.sort(
            key=lambda idxs: (
                sum(node.gpu_list[i].free_milli - pod.gpu_milli for i in idxs),
                idxs,
            )
        )
        self._placement_cache[cache_key] = out
        return out

    def feasible_nodes(self, pod: Pod) -> List[FeasibleNode]:
        feasible: List[FeasibleNode] = []
        for node in self.nodes:
            if node.cpu_avail < pod.cpu_milli:
                continue
            if node.memory_avail < pod.memory_mib:
                continue
            if not self._can_match_model(node, pod):
                continue
            candidates = self._candidate_gpu_sets(node, pod)
            if not candidates:
                continue
            feasible.append(
                FeasibleNode(
                    node_id=node.node_id,
                    gpu_candidates=candidates,
                    node_ref=node,
                )
            )
        feasible.sort(key=lambda x: x.node_id)
        return feasible

    def prepare_next_decision(self) -> DecisionContext:
        if self.done:
            raise RuntimeError("simulation is already done")
        if self._prepared is not None:
            return self._prepared

        pod = self.pods[self.incoming_index]
        self.current_time = pod.creation_time
        self._process_completions()

        self.pending_pods.append(pod.name)
        self.pending_pods.sort(
            key=lambda name: (
                self.pod_map[name].creation_time,
                -self.pod_map[name].priority,
                name,
            )
        )

        current_name = self.pending_pods[0]
        current_pod = self.pod_map[current_name]
        feasible = self.feasible_nodes(current_pod)

        self.arrived_gpu_milli_sum += current_pod.total_gpu_milli
        self._prepared = DecisionContext(
            current_time=self.current_time,
            pod=current_pod,
            feasible_nodes=feasible,
        )
        return self._prepared

    def simulate_node_after(self, node: Node, pod: Pod, gpu_indices: Sequence[int]) -> Optional[Node]:
        if node.cpu_avail < pod.cpu_milli or node.memory_avail < pod.memory_mib:
            return None
        if len(gpu_indices) != pod.num_gpu:
            return None
        if len(set(gpu_indices)) != len(gpu_indices):
            return None

        clone = copy.deepcopy(node)
        clone.cpu_avail -= pod.cpu_milli
        clone.memory_avail -= pod.memory_mib
        for idx in gpu_indices:
            if idx < 0 or idx >= clone.gpu_count:
                return None
            if clone.gpu_list[idx].free_milli < pod.gpu_milli:
                return None
            clone.gpu_list[idx].allocated_milli += pod.gpu_milli
        return clone

    def _can_place_demand(self, free_milli: Sequence[int], demand: Tuple[int, int]) -> bool:
        n, m = demand
        return sum(1 for x in free_milli if x >= m) >= n

    def _utilization_fragmentation_score(self, node: Node) -> float:
        if node.gpu_count == 0:
            return 0.0
        total = float(node.gpu_count * 1000)
        free = float(sum(g.free_milli for g in node.gpu_list))
        return max(0.0, min(1.0, 1.0 - (free / total)))

    def fragmentation_score(self, node: Node) -> float:
        if node.gpu_count == 0:
            return 0.0
        key = (node.model, tuple(g.free_milli for g in node.gpu_list))
        cached = self._frag_cache.get(key)
        if cached is not None:
            return cached

        if self.fragmentation_mode == "utilization":
            score = self._utilization_fragmentation_score(node)
        else:
            dist = self.demand_distribution.get(node.model, {})
            if not dist:
                self._frag_cache[key] = 0.0
                return 0.0

            free = [g.free_milli for g in node.gpu_list]
            placeable = 0.0
            for demand, prob in dist.items():
                if self._can_place_demand(free, demand):
                    placeable += prob

            score = 1.0 - placeable

        score *= self.fragmentation_scale
        self._frag_cache[key] = score
        return score

    def _cluster_fragmentation_score(self) -> float:
        gpu_nodes = [n for n in self.nodes if n.gpu_count > 0]
        if not gpu_nodes:
            return 0.0
        weighted = 0.0
        total_gpu = 0
        for node in gpu_nodes:
            weighted += self.fragmentation_score(node) * node.gpu_count
            total_gpu += node.gpu_count
        return weighted / total_gpu if total_gpu > 0 else 0.0

    def _apply_action(
        self,
        context: DecisionContext,
        action: Union[None, str, Dict[str, object], Tuple[object, ...]],
    ) -> Tuple[bool, str, Optional[str], Tuple[int, ...]]:
        pod = context.pod
        feasible_by_node = {x.node_id: x for x in context.feasible_nodes}

        if not context.feasible_nodes:
            return False, "no_feasible_node", None, ()
        if action is None:
            return False, "rejected", None, ()

        node_id: Optional[str] = None
        gpu_indices: Optional[Tuple[int, ...]] = None

        if isinstance(action, str):
            node_id = action
        elif isinstance(action, tuple) and len(action) >= 1:
            node_id = str(action[0])
            if len(action) >= 2:
                gpu_indices = tuple(int(x) for x in action[1])
        elif isinstance(action, dict):
            node_id_raw = action.get("node_id")
            node_id = str(node_id_raw) if node_id_raw is not None else None
            gpu_raw = action.get("gpu_indices")
            if gpu_raw is not None:
                gpu_indices = tuple(int(x) for x in gpu_raw)

        if not node_id or node_id not in feasible_by_node:
            return False, "invalid_action", None, ()

        item = feasible_by_node[node_id]
        if gpu_indices is None:
            gpu_indices = item.gpu_candidates[0] if item.gpu_candidates else ()

        if gpu_indices not in item.gpu_candidates:
            return False, "invalid_gpu_choice", None, ()

        node = self.node_map[node_id]
        node.cpu_avail -= pod.cpu_milli
        node.memory_avail -= pod.memory_mib
        for idx in gpu_indices:
            node.gpu_list[idx].allocated_milli += pod.gpu_milli

        pod.scheduled_time = self.current_time
        pod.assigned_node = node_id
        pod.assigned_gpus = gpu_indices

        running = RunningPod(
            pod_name=pod.name,
            node_id=node_id,
            gpu_indices=gpu_indices,
            cpu_milli=pod.cpu_milli,
            memory_mib=pod.memory_mib,
            gpu_milli=pod.gpu_milli,
            deletion_time=pod.deletion_time,
            creation_time=pod.creation_time,
            scheduled_time=self.current_time,
        )
        self.running_map[pod.name] = running
        heapq.heappush(self.running_heap, (running.deletion_time, self._running_seq, pod.name))
        self._running_seq += 1

        self.allocated_gpu_milli_sum += pod.total_gpu_milli
        return True, "scheduled", node_id, gpu_indices

    def _record_event_metrics(self, scheduled: bool) -> Dict[str, float]:
        used_gpu_milli = sum(n.used_gpu_milli for n in self.nodes)
        gar = (
            0.0
            if self.total_gpu_capacity_milli <= 0
            else self.allocated_gpu_milli_sum / self.total_gpu_capacity_milli
        )
        metrics = {
            "current_time": float(self.current_time),
            "scheduled": 1.0 if scheduled else 0.0,
            "used_gpu_milli": float(used_gpu_milli),
            "arrived_gpu_milli_sum": float(self.arrived_gpu_milli_sum),
            "allocated_gpu_milli_sum": float(self.allocated_gpu_milli_sum),
            "gpu_allocation_ratio": float(gar),
            "unallocated_gpu_fraction": float(1.0 - gar),
            "failed_count": float(len(self.failed)),
            "cluster_fragmentation_score": float(self._cluster_fragmentation_score()),
        }
        if self.record_history:
            self.event_metrics.append(metrics)
        return metrics

    def step(
        self,
        action: Union[None, str, Dict[str, object], Tuple[object, ...]],
    ) -> StepResult:
        context = self.prepare_next_decision()
        pod = context.pod

        if self.pending_pods and self.pending_pods[0] == pod.name:
            self.pending_pods.pop(0)
        elif pod.name in self.pending_pods:
            self.pending_pods.remove(pod.name)

        scheduled, reason, node_id, gpu_indices = self._apply_action(context, action)
        if not scheduled:
            self.failed.append(pod.name)

        self.incoming_index += 1
        self.done = self.incoming_index >= len(self.pods)
        self._prepared = None

        metrics = self._record_event_metrics(scheduled)
        return StepResult(
            pod_name=pod.name,
            current_time=self.current_time,
            scheduled=scheduled,
            failed=not scheduled,
            reason=reason,
            chosen_node=node_id,
            chosen_gpus=gpu_indices,
            feasible_node_ids=[x.node_id for x in context.feasible_nodes],
            metrics=metrics,
            done=self.done,
        )

    def run_policy(self, policy: SchedulerPolicy) -> Dict[str, object]:
        self.reset()

        while not self.done:
            context = self.prepare_next_decision()
            action = policy.choose_action(context, self)
            self.step(action)

        while self.running_heap:
            next_end = self.running_heap[0][0]
            self.current_time = next_end
            self._process_completions()

        waits = [x.scheduled_time - x.creation_time for x in self.completed]
        jcts = [x.deletion_time - x.creation_time for x in self.completed]
        makespan = max((x.deletion_time for x in self.completed), default=self.current_time)
        final_fragmentation = self._cluster_fragmentation_score()

        summary: Dict[str, object] = {
            "policy": policy.name,
            "total_pods": len(self.pods),
            "scheduled_pods": len(self.completed),
            "failed_pods": len(self.failed),
            "total_gpu_milli_capacity": self.total_gpu_capacity_milli,
            "allocated_gpu_milli_sum": self.allocated_gpu_milli_sum,
            "gpu_allocation_ratio": (
                0.0
                if self.total_gpu_capacity_milli <= 0
                else self.allocated_gpu_milli_sum / self.total_gpu_capacity_milli
            ),
            "unallocated_gpu_fraction": (
                1.0
                if self.total_gpu_capacity_milli <= 0
                else 1.0 - (self.allocated_gpu_milli_sum / self.total_gpu_capacity_milli)
            ),
            "avg_wait_time": float(mean(waits)) if waits else 0.0,
            "avg_jct": float(mean(jcts)) if jcts else 0.0,
            "makespan": makespan,
            "cluster_fragmentation_score": final_fragmentation,
        }
        if self.record_history:
            summary["event_metrics"] = self.event_metrics
        return summary


def validate_policies(
    *,
    node_csv: Path,
    pod_csv: Path,
    policy_names: Sequence[str],
    seed: int = 42,
    pod_limit: Optional[int] = None,
    drop_missing_deletion: bool = False,
    default_duration: int = 3600,
    min_pod_duration: int = 0,
    record_history: bool = False,
) -> Dict[str, object]:
    nodes = load_nodes(node_csv)
    pods = load_pods(
        pod_csv,
        drop_missing_deletion=drop_missing_deletion,
        default_duration=default_duration,
        min_pod_duration=min_pod_duration,
        limit=pod_limit,
    )
    models = [n.model for n in nodes if n.gpu_count > 0]
    demand_distribution = build_gpu_demand_distribution(pods, models)

    out: Dict[str, object] = {
        "node_csv": str(node_csv),
        "pod_csv": str(pod_csv),
        "num_nodes": len(nodes),
        "num_pods": len(pods),
        "policies": {},
    }

    sim = ClusterSimulator(
        nodes,
        pods,
        demand_distribution=demand_distribution,
        record_history=record_history,
    )

    for name in policy_names:
        policy = policy_from_name(name, seed=seed)
        out["policies"][policy.name] = sim.run_policy(policy)

    policies = out["policies"]
    if "fgd" in policies:
        fgd_unalloc = policies["fgd"]["unallocated_gpu_fraction"]
        rel = {}
        for key, val in policies.items():
            if key == "fgd":
                continue
            baseline_unalloc = val["unallocated_gpu_fraction"]
            if baseline_unalloc <= 0:
                rel[key] = 0.0
            else:
                rel[key] = (baseline_unalloc - fgd_unalloc) / baseline_unalloc
        out["fgd_relative_unallocated_reduction"] = rel

    return out


def _default_csv_paths() -> Tuple[Path, Path]:
    base = Path(__file__).resolve().parent / "clusterdata" / "cluster-trace-gpu-v2023" / "csv"
    return base / "openb_node_list_gpu_node.csv", base / "openb_pod_list_default.csv"


def main() -> None:
    default_node, default_pod = _default_csv_paths()
    parser = argparse.ArgumentParser(description="Discrete-event GPU scheduler simulator")
    parser.add_argument("--node-csv", type=Path, default=default_node)
    parser.add_argument("--pod-csv", type=Path, default=default_pod)
    parser.add_argument("--policy", type=str, default="fgd")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pod-limit", type=int, default=None)
    parser.add_argument("--drop-missing-deletion", action="store_true")
    parser.add_argument("--default-duration", type=int, default=3600)
    parser.add_argument("--min-pod-duration", type=int, default=600000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--record-history", action="store_true")
    args = parser.parse_args()

    if args.validate:
        result = validate_policies(
            node_csv=args.node_csv,
            pod_csv=args.pod_csv,
            policy_names=[
                "random",
                "best_fit",
                "dot_product",
                "gpu_packing",
                "gpu_clustering",
                "fgd",
            ],
            seed=args.seed,
            pod_limit=args.pod_limit,
            drop_missing_deletion=args.drop_missing_deletion,
            default_duration=args.default_duration,
            min_pod_duration=args.min_pod_duration,
            record_history=args.record_history,
        )
    else:
        result = validate_policies(
            node_csv=args.node_csv,
            pod_csv=args.pod_csv,
            policy_names=[args.policy],
            seed=args.seed,
            pod_limit=args.pod_limit,
            drop_missing_deletion=args.drop_missing_deletion,
            default_duration=args.default_duration,
            min_pod_duration=args.min_pod_duration,
            record_history=args.record_history,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
