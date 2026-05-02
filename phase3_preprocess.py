from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


NODE_COLUMNS = ("sn", "cpu_milli", "memory_mib", "gpu", "model")
POD_COLUMNS = (
    "name",
    "cpu_milli",
    "memory_mib",
    "num_gpu",
    "gpu_milli",
    "gpu_spec",
    "qos",
    "creation_time",
    "deletion_time",
    "scheduled_time",
)

KNOWN_GPU_MODELS = {
    "P100",
    "V100M16",
    "V100M32",
    "T4",
    "A10",
    "G1",
    "G2",
    "G3",
}


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_gpu_spec(raw: object) -> Tuple[str, ...]:
    if raw is None:
        return ()
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ()
    parts = [p.strip() for p in s.split("|") if p.strip()]
    seen = set()
    out: List[str] = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            out.append(part)
    return tuple(out)


def _normalize_qos(raw_qos: object) -> Tuple[str, bool]:
    s = "" if raw_qos is None else str(raw_qos).strip()
    if not s:
        return "BestEffort", True

    lowered = s.lower()
    if lowered == "guaranteed":
        return "Guaranteed", s != "Guaranteed"
    if lowered == "burstable":
        return "Burstable", s != "Burstable"
    if lowered in {"be", "besteffort", "best-effort", "best_effort"}:
        return "BestEffort", s != "BestEffort"
    if lowered == "ls":
        return "LS", s != "LS"
    return "BestEffort", True


def _assign_priority(qos: str, num_gpu: int, gpu_milli: int) -> int:
    if qos == "Guaranteed" and 0 < gpu_milli < 1000:
        return 1
    if qos in {"Burstable", "BestEffort"} and num_gpu >= 4:
        return 0
    return 0


def _read_csv(path: Path, expected_columns: Sequence[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        missing = [c for c in expected_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def _compute_quantiles(values: Sequence[int], quantiles: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {f"q{int(q * 100)}": 0.0 for q in quantiles}
    sorted_values = sorted(values)
    n = len(sorted_values)
    out: Dict[str, float] = {}
    for q in quantiles:
        if q <= 0:
            out[f"q{int(q * 100)}"] = float(sorted_values[0])
            continue
        if q >= 1:
            out[f"q{int(q * 100)}"] = float(sorted_values[-1])
            continue
        idx = min(n - 1, max(0, int(math.ceil(q * n) - 1)))
        out[f"q{int(q * 100)}"] = float(sorted_values[idx])
    return out


def clean_nodes(raw_nodes: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, object]], Dict[str, int], Dict[str, object]]:
    report = Counter()
    cleaned: List[Dict[str, object]] = []
    seen_sn = set()

    for row in raw_nodes:
        sn = str(row.get("sn", "")).strip()
        if not sn:
            report["missing_sn"] += 1
            continue
        if sn in seen_sn:
            report["duplicate_sn"] += 1
            continue
        seen_sn.add(sn)

        cpu_milli = _to_int(row.get("cpu_milli"))
        memory_mib = _to_int(row.get("memory_mib"))
        gpu = _to_int(row.get("gpu"))

        if cpu_milli is None or memory_mib is None or cpu_milli <= 0 or memory_mib <= 0:
            report["nonpositive_cpu_or_memory"] += 1
            continue
        if gpu is None or gpu < 0:
            report["invalid_gpu_count"] += 1
            continue

        model = str(row.get("model", "")).strip()
        if gpu == 0:
            model = "CPU-only"
        elif model not in KNOWN_GPU_MODELS:
            report["unknown_gpu_model"] += 1
            continue

        cleaned.append(
            {
                "sn": sn,
                "cpu_milli": cpu_milli,
                "memory_mib": memory_mib,
                "gpu": gpu,
                "model": model,
                "is_gpu_node": 1 if gpu > 0 else 0,
            }
        )

    for idx, node in enumerate(cleaned):
        node["node_id"] = idx

    gpu_models_present = sorted({str(n["model"]) for n in cleaned if int(n["gpu"]) > 0})
    max_cpu = max((int(n["cpu_milli"]) for n in cleaned), default=0)
    max_mem = max((int(n["memory_mib"]) for n in cleaned), default=0)
    max_gpu = max((int(n["gpu"]) for n in cleaned), default=0)
    total_gpu_capacity_milli = sum(int(n["gpu"]) * 1000 for n in cleaned)

    cluster_caps: Dict[str, object] = {
        "gpu_models_present": gpu_models_present,
        "max_cpu_milli_per_node": max_cpu,
        "max_memory_mib_per_node": max_mem,
        "max_gpu_per_node": max_gpu,
        "total_gpu_capacity_milli": total_gpu_capacity_milli,
        "num_gpu_nodes": sum(1 for n in cleaned if int(n["gpu"]) > 0),
        "num_cpu_only_nodes": sum(1 for n in cleaned if int(n["gpu"]) == 0),
    }

    return cleaned, dict(report), cluster_caps


def clean_pods(
    raw_pods: Sequence[Dict[str, str]],
    cluster_caps: Dict[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, int]], List[str]]:
    gpu_models_present = set(cluster_caps["gpu_models_present"])
    max_cpu = int(cluster_caps["max_cpu_milli_per_node"])
    max_mem = int(cluster_caps["max_memory_mib_per_node"])
    max_gpu = int(cluster_caps["max_gpu_per_node"])

    removed_reasons = Counter()
    adjusted_reasons = Counter()
    cleaned: List[Dict[str, object]] = []

    for row_idx, row in enumerate(raw_pods):
        creation_time = _to_int(row.get("creation_time"))
        deletion_time = _to_int(row.get("deletion_time"))
        if creation_time is None or deletion_time is None:
            removed_reasons["missing_lifecycle_timestamp"] += 1
            continue
        if deletion_time <= creation_time:
            removed_reasons["invalid_lifecycle_order"] += 1
            continue

        cpu_milli = _to_int(row.get("cpu_milli"))
        memory_mib = _to_int(row.get("memory_mib"))
        if cpu_milli is None or memory_mib is None or cpu_milli <= 0 or memory_mib <= 0:
            removed_reasons["nonpositive_cpu_or_memory"] += 1
            continue

        num_gpu = _to_int(row.get("num_gpu"))
        if num_gpu is None or num_gpu < 0:
            removed_reasons["invalid_num_gpu"] += 1
            continue
        if num_gpu > max_gpu:
            removed_reasons["num_gpu_exceeds_cluster_max"] += 1
            continue

        if cpu_milli > max_cpu or memory_mib > max_mem:
            removed_reasons["resource_exceeds_any_node_capacity"] += 1
            continue

        gpu_milli_raw = _to_int(row.get("gpu_milli"))
        if num_gpu == 0:
            if gpu_milli_raw is not None and gpu_milli_raw > 0:
                removed_reasons["gpu_milli_positive_with_zero_num_gpu"] += 1
                continue
            gpu_milli = 0
        else:
            if gpu_milli_raw is None or gpu_milli_raw <= 0:
                gpu_milli = 1000
                adjusted_reasons["gpu_milli_imputed_to_1000_for_gpu_pod"] += 1
            elif gpu_milli_raw > 1000:
                gpu_milli = 1000
                adjusted_reasons["gpu_milli_clamped_to_1000"] += 1
            else:
                gpu_milli = gpu_milli_raw

        raw_spec = _parse_gpu_spec(row.get("gpu_spec"))
        filtered_spec = tuple(spec for spec in raw_spec if spec in gpu_models_present)
        if raw_spec and len(filtered_spec) != len(raw_spec):
            adjusted_reasons["gpu_spec_unknown_models_removed"] += 1

        if num_gpu > 0 and raw_spec and not filtered_spec:
            removed_reasons["gpu_spec_unavailable_in_cluster"] += 1
            continue

        qos, qos_changed = _normalize_qos(row.get("qos"))
        if qos_changed:
            adjusted_reasons["qos_normalized_or_defaulted"] += 1

        scheduled_time = _to_int(row.get("scheduled_time"))
        output_row: Dict[str, object] = dict(row)
        output_row["cpu_milli"] = cpu_milli
        output_row["memory_mib"] = memory_mib
        output_row["num_gpu"] = num_gpu
        output_row["gpu_milli"] = gpu_milli
        output_row["gpu_spec"] = "|".join(filtered_spec)
        output_row["qos"] = qos
        output_row["creation_time"] = creation_time
        output_row["deletion_time"] = deletion_time
        output_row["scheduled_time"] = "" if scheduled_time is None else scheduled_time
        output_row["priority"] = _assign_priority(qos=qos, num_gpu=num_gpu, gpu_milli=gpu_milli)
        output_row["source_row_index"] = row_idx

        cleaned.append(output_row)

    cleaned.sort(key=lambda r: (int(r["creation_time"]), int(r["source_row_index"])))
    for idx, pod in enumerate(cleaned):
        pod["pod_index"] = idx

    monotonic_ok = all(
        int(cleaned[i]["creation_time"]) <= int(cleaned[i + 1]["creation_time"])
        for i in range(max(0, len(cleaned) - 1))
    )
    if not monotonic_ok:
        raise RuntimeError("Pod creation_time ordering is not monotonic after sorting")

    pod_report = {
        "removed": dict(removed_reasons),
        "adjusted": dict(adjusted_reasons),
    }
    derived_columns = ["priority", "source_row_index", "pod_index"]
    return cleaned, pod_report, derived_columns


def temporal_split(
    pods: Sequence[Dict[str, object]],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    total = len(pods)
    train_end = int(math.floor(train_ratio * total))
    val_end = train_end + int(math.floor(val_ratio * total))
    train = list(pods[:train_end])
    val = list(pods[train_end:val_end])
    test = list(pods[val_end:])
    return train, val, test


def build_episodes(
    pods: Sequence[Dict[str, object]],
    episode_len: int,
    split_name: str,
) -> Tuple[List[Dict[str, object]], int]:
    if episode_len <= 0:
        raise ValueError("episode_len must be > 0")

    episodes: List[Dict[str, object]] = []
    usable = len(pods) - (len(pods) % episode_len)
    discarded = len(pods) - usable

    for start in range(0, usable, episode_len):
        end = start + episode_len - 1
        first = pods[start]
        last = pods[end]
        episodes.append(
            {
                "episode_id": f"{split_name}_{len(episodes)}",
                "split": split_name,
                "start_idx": start,
                "end_idx": end,
                "length": episode_len,
                "start_creation_time": int(first["creation_time"]),
                "end_creation_time": int(last["creation_time"]),
                "start_pod_name": str(first.get("name", "")),
                "end_pod_name": str(last.get("name", "")),
            }
        )

    return episodes, discarded


def _bucket_gpu_milli(value: int, bin_size: int) -> int:
    if bin_size <= 0:
        return value
    if value <= 0:
        return 0
    return int(math.ceil(value / bin_size) * bin_size)


def build_demand_distribution(
    train_pods: Sequence[Dict[str, object]],
    gpu_models: Sequence[str],
    gpu_milli_bin_size: int,
) -> Dict[str, object]:
    counts_by_model: Dict[str, Counter[Tuple[int, int]]] = {
        model: Counter() for model in sorted(gpu_models)
    }

    for pod in train_pods:
        num_gpu = int(pod["num_gpu"])
        if num_gpu <= 0:
            continue

        gpu_milli = _bucket_gpu_milli(int(pod["gpu_milli"]), gpu_milli_bin_size)
        spec = _parse_gpu_spec(pod.get("gpu_spec"))

        if spec:
            target_models = [m for m in counts_by_model if m in spec]
        else:
            target_models = list(counts_by_model.keys())

        demand = (num_gpu, gpu_milli)
        for model in target_models:
            counts_by_model[model][demand] += 1

    distribution: Dict[str, Dict[str, float]] = {}
    detailed: Dict[str, List[Dict[str, object]]] = {}

    for model in sorted(counts_by_model.keys()):
        model_counts = counts_by_model[model]
        total = sum(model_counts.values())
        if total == 0:
            distribution[model] = {}
            detailed[model] = []
            continue

        model_dist: Dict[str, float] = {}
        model_details: List[Dict[str, object]] = []
        for (num_gpu, gpu_milli), count in sorted(model_counts.items()):
            key = f"{num_gpu}|{gpu_milli}"
            prob = count / total
            model_dist[key] = prob
            model_details.append(
                {
                    "num_gpu": num_gpu,
                    "gpu_milli": gpu_milli,
                    "count": count,
                    "probability": prob,
                }
            )

        distribution[model] = model_dist
        detailed[model] = model_details

    return {
        "gpu_milli_bin_size": gpu_milli_bin_size,
        "distribution": distribution,
        "detailed": detailed,
    }


def _slice_inter_arrival_gaps(pods: Sequence[Dict[str, object]]) -> List[int]:
    if len(pods) < 2:
        return []
    out = []
    for i in range(1, len(pods)):
        gap = int(pods[i]["creation_time"]) - int(pods[i - 1]["creation_time"])
        out.append(gap)
    return out


def _split_stats(name: str, pods: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not pods:
        return {
            "split": name,
            "pods": 0,
            "gpu_pods": 0,
            "cpu_only_pods": 0,
            "sum_gpu_milli_requested": 0,
            "creation_time_start": None,
            "creation_time_end": None,
        }

    gaps = _slice_inter_arrival_gaps(pods)
    gap_quantiles = _compute_quantiles(gaps, [0.5, 0.9, 0.99]) if gaps else {}
    return {
        "split": name,
        "pods": len(pods),
        "gpu_pods": sum(1 for p in pods if int(p["num_gpu"]) > 0),
        "cpu_only_pods": sum(1 for p in pods if int(p["num_gpu"]) == 0),
        "sum_gpu_milli_requested": sum(int(p["num_gpu"]) * int(p["gpu_milli"]) for p in pods),
        "creation_time_start": int(pods[0]["creation_time"]),
        "creation_time_end": int(pods[-1]["creation_time"]),
        "inter_arrival_gap_ms": {
            "min": min(gaps) if gaps else 0,
            "max": max(gaps) if gaps else 0,
            **gap_quantiles,
        },
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _top_demands(demand_distribution: Dict[str, object], top_k: int = 10) -> Dict[str, List[Dict[str, object]]]:
    detailed = demand_distribution["detailed"]
    out: Dict[str, List[Dict[str, object]]] = {}
    for model, demands in detailed.items():
        sorted_demands = sorted(demands, key=lambda d: (-float(d["probability"]), int(d["num_gpu"]), int(d["gpu_milli"])))
        out[model] = sorted_demands[:top_k]
    return out


def preprocess(
    node_csv: Path,
    pod_csv: Path,
    output_dir: Path,
    episode_len: int,
    train_ratio: float,
    val_ratio: float,
    gpu_milli_bin_size: int,
) -> Dict[str, object]:
    raw_nodes, node_fieldnames = _read_csv(node_csv, expected_columns=NODE_COLUMNS)
    raw_pods, pod_fieldnames = _read_csv(pod_csv, expected_columns=POD_COLUMNS)

    nodes_clean, node_removed_reasons, cluster_caps = clean_nodes(raw_nodes)
    pods_clean, pod_report, derived_pod_columns = clean_pods(raw_pods, cluster_caps=cluster_caps)

    train_pods, val_pods, test_pods = temporal_split(
        pods_clean,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_episodes, train_tail = build_episodes(train_pods, episode_len=episode_len, split_name="train")
    val_episodes, val_tail = build_episodes(val_pods, episode_len=episode_len, split_name="val")

    demand_distribution = build_demand_distribution(
        train_pods=train_pods,
        gpu_models=cluster_caps["gpu_models_present"],
        gpu_milli_bin_size=gpu_milli_bin_size,
    )

    node_output_fields = ["node_id", *node_fieldnames, "is_gpu_node"]
    pod_output_fields = [*pod_fieldnames, *derived_pod_columns]

    _write_csv(output_dir / "nodes_clean.csv", nodes_clean, node_output_fields)
    _write_csv(output_dir / "pods_train.csv", train_pods, pod_output_fields)
    _write_csv(output_dir / "pods_val.csv", val_pods, pod_output_fields)
    _write_csv(output_dir / "pods_test.csv", test_pods, pod_output_fields)

    episode_fields = [
        "episode_id",
        "split",
        "start_idx",
        "end_idx",
        "length",
        "start_creation_time",
        "end_creation_time",
        "start_pod_name",
        "end_pod_name",
    ]
    _write_csv(output_dir / "episodes_train.csv", train_episodes, episode_fields)
    _write_csv(output_dir / "episodes_val.csv", val_episodes, episode_fields)

    with (output_dir / "workload_demand_distribution.json").open("w") as f:
        json.dump(demand_distribution, f, indent=2, sort_keys=True)

    all_gaps = _slice_inter_arrival_gaps(pods_clean)
    all_gap_quantiles = _compute_quantiles(all_gaps, [0.5, 0.9, 0.99]) if all_gaps else {}
    total_gpu_demand_milli = sum(int(p["num_gpu"]) * int(p["gpu_milli"]) for p in pods_clean)

    cleaning_report: Dict[str, object] = {
        "input": {
            "node_csv": str(node_csv),
            "pod_csv": str(pod_csv),
            "nodes_raw": len(raw_nodes),
            "pods_raw": len(raw_pods),
        },
        "nodes": {
            "kept": len(nodes_clean),
            "removed": len(raw_nodes) - len(nodes_clean),
            "removed_reasons": node_removed_reasons,
            "cluster_caps": cluster_caps,
        },
        "pods": {
            "kept": len(pods_clean),
            "removed": len(raw_pods) - len(pods_clean),
            "removed_reasons": pod_report["removed"],
            "adjustments": pod_report["adjusted"],
            "priority_counts": dict(Counter(int(p["priority"]) for p in pods_clean)),
        },
        "split": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 1.0 - train_ratio - val_ratio,
            "train": _split_stats("train", train_pods),
            "val": _split_stats("val", val_pods),
            "test": _split_stats("test", test_pods),
        },
        "episodes": {
            "episode_len": episode_len,
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes),
            "train_tail_discarded": train_tail,
            "val_tail_discarded": val_tail,
        },
        "sanity_checks": {
            "creation_time_monotonic": True,
            "inter_arrival_gap_ms": {
                "count": len(all_gaps),
                "min": min(all_gaps) if all_gaps else 0,
                "max": max(all_gaps) if all_gaps else 0,
                **all_gap_quantiles,
            },
            "gpu_capacity_vs_demand": {
                "total_gpu_capacity_milli": int(cluster_caps["total_gpu_capacity_milli"]),
                "total_gpu_requested_milli": total_gpu_demand_milli,
                "requested_over_capacity_ratio": (
                    0.0
                    if int(cluster_caps["total_gpu_capacity_milli"]) == 0
                    else total_gpu_demand_milli / int(cluster_caps["total_gpu_capacity_milli"])
                ),
            },
            "top_demand_types_per_model": _top_demands(demand_distribution),
        },
    }

    with (output_dir / "cleaning_report.json").open("w") as f:
        json.dump(cleaning_report, f, indent=2, sort_keys=True)

    return cleaning_report


def _default_csv_paths() -> Tuple[Path, Path]:
    base = Path(__file__).resolve().parent / "clusterdata" / "cluster-trace-gpu-v2023" / "csv"
    return base / "openb_node_list_all_node.csv", base / "openb_pod_list_default.csv"


def main() -> None:
    default_node, default_pod = _default_csv_paths()

    parser = argparse.ArgumentParser(description="Phase 3 deterministic data preprocessing")
    parser.add_argument("--node-csv", type=Path, default=default_node)
    parser.add_argument("--pod-csv", type=Path, default=default_pod)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "preprocessed" / "phase3",
    )
    parser.add_argument("--episode-len", type=int, default=500)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--gpu-milli-bin-size",
        type=int,
        default=0,
        help="0 keeps raw gpu_milli; positive value buckets by bin upper bound",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be > 0 and train_ratio + val_ratio < 1")

    report = preprocess(
        node_csv=args.node_csv,
        pod_csv=args.pod_csv,
        output_dir=args.output_dir,
        episode_len=args.episode_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        gpu_milli_bin_size=args.gpu_milli_bin_size,
    )
    print(json.dumps(
        {
            "nodes_kept": report["nodes"]["kept"],
            "pods_kept": report["pods"]["kept"],
            "train_pods": report["split"]["train"]["pods"],
            "val_pods": report["split"]["val"]["pods"],
            "test_pods": report["split"]["test"]["pods"],
            "train_episodes": report["episodes"]["train_episodes"],
            "val_episodes": report["episodes"]["val_episodes"],
            "output_dir": str(args.output_dir),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
