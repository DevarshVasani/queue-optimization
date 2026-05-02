from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float
    wall_time: Optional[float] = None


@dataclass
class ScalarSeries:
    name: str
    points: List[ScalarPoint]

    def sorted(self) -> "ScalarSeries":
        ordered = sorted(self.points, key=lambda item: (item.step, item.wall_time or 0.0))
        deduped: Dict[int, ScalarPoint] = {}
        for point in ordered:
            deduped[int(point.step)] = point
        return ScalarSeries(name=self.name, points=[deduped[key] for key in sorted(deduped)])

    @property
    def steps(self) -> np.ndarray:
        return np.asarray([point.step for point in self.points], dtype=float)

    @property
    def values(self) -> np.ndarray:
        return np.asarray([point.value for point in self.points], dtype=float)


TensorboardRunMap = Dict[str, Dict[str, ScalarSeries]]

DEFAULT_1A_TAGS: Tuple[Tuple[str, str, Sequence[str]], ...] = (
    ("episode_reward", "Episode reward", ("rollout/ep_rew_mean", "episode/reward", "train/episode_reward")),
    ("validation_gar", "Validation GAR", ("validation/gar", "episode/gar")),
    ("validation_success_rate", "Validation success rate", ("validation/success_rate", "episode/success_rate")),
    ("validation_latency_objective", "Validation latency objective", ("validation/latency_objective", "episode/latency_objective")),
    ("policy_entropy", "Policy entropy", ("train/entropy_loss", "train/policy_entropy", "rollout/policy_entropy")),
)

DEFAULT_1A_PANEL_LOOKUP: Dict[str, Tuple[str, Sequence[str]]] = {
    key: (label, tags) for key, label, tags in DEFAULT_1A_TAGS
}


def discover_event_files(log_root: Path) -> List[Path]:
    if not log_root.exists():
        raise FileNotFoundError(f"log root does not exist: {log_root}")
    if log_root.is_file():
        return [log_root]
    return sorted(path for path in log_root.rglob("events.out.tfevents*") if path.is_file())


def _run_name_for_event_file(log_root: Path, event_file: Path) -> str:
    parent = event_file.parent
    try:
        relative = parent.relative_to(log_root)
        return str(relative).replace("\\", "/") or parent.name
    except ValueError:
        return parent.name


def _load_tensorboard_event_file(event_file: Path) -> Dict[str, List[ScalarPoint]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:  # pragma: no cover - exercised in smoke tests via synthetic series
        raise ImportError(
            "TensorBoard is required to read training event files. Install tensorboard or use CSV inputs."
        ) from exc

    accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()

    scalars: Dict[str, List[ScalarPoint]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        points: List[ScalarPoint] = []
        for event in accumulator.Scalars(tag):
            points.append(ScalarPoint(step=int(event.step), value=float(event.value), wall_time=float(event.wall_time)))
        if points:
            scalars[tag] = points
    return scalars


def load_tensorboard_runs(log_root: Path) -> TensorboardRunMap:
    event_files = discover_event_files(log_root)
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event files found under {log_root}")

    runs: TensorboardRunMap = {}
    for event_file in event_files:
        run_name = _run_name_for_event_file(log_root, event_file)
        scalars = _load_tensorboard_event_file(event_file)
        run_bucket = runs.setdefault(run_name, {})
        for tag, points in scalars.items():
            run_bucket.setdefault(tag, ScalarSeries(name=tag, points=[])).points.extend(points)

    for run_name, tag_map in list(runs.items()):
        for tag, series in list(tag_map.items()):
            tag_map[tag] = series.sorted()
        runs[run_name] = dict(sorted(tag_map.items(), key=lambda item: item[0]))
    return dict(sorted(runs.items(), key=lambda item: item[0]))


def build_series(points: Sequence[Tuple[int, float]], *, name: str = "series") -> ScalarSeries:
    return ScalarSeries(name=name, points=[ScalarPoint(step=int(step), value=float(value)) for step, value in points]).sorted()


def smooth_values(values: Sequence[float], window: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or window <= 1:
        return array.copy()
    window = min(int(window), array.size)
    kernel = np.ones(window, dtype=float) / float(window)
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded = np.pad(array, (left_pad, right_pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def interpolate_series(series: ScalarSeries, step_grid: np.ndarray) -> np.ndarray:
    if not series.points:
        return np.full(step_grid.shape, np.nan, dtype=float)
    ordered = series.sorted()
    steps = ordered.steps
    values = ordered.values
    if steps.size == 1:
        out = np.full(step_grid.shape, values[0], dtype=float)
        out[(step_grid < steps[0]) | (step_grid > steps[0])] = np.nan
        return out
    interpolated = np.interp(step_grid, steps, values)
    interpolated[step_grid < steps[0]] = np.nan
    interpolated[step_grid > steps[-1]] = np.nan
    return interpolated


def common_step_grid(series_list: Sequence[ScalarSeries], *, max_points: int = 250) -> np.ndarray:
    all_steps: List[int] = []
    for series in series_list:
        all_steps.extend(int(point.step) for point in series.points)
    if not all_steps:
        return np.asarray([], dtype=float)
    unique_steps = np.asarray(sorted(set(all_steps)), dtype=float)
    if unique_steps.size <= max_points:
        return unique_steps
    start = float(unique_steps[0])
    end = float(unique_steps[-1])
    return np.linspace(start, end, max_points, dtype=float)


def aggregate_series(series_list: Sequence[ScalarSeries], *, max_points: int = 250) -> Dict[str, np.ndarray]:
    grid = common_step_grid(series_list, max_points=max_points)
    if grid.size == 0:
        return {"steps": grid, "mean": np.asarray([], dtype=float), "std": np.asarray([], dtype=float)}

    matrix = np.vstack([interpolate_series(series, grid) for series in series_list])
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    return {"steps": grid, "mean": mean, "std": std, "matrix": matrix}


def first_crossing_step(series: ScalarSeries, target: float, *, higher_is_better: bool = True) -> Optional[int]:
    ordered = series.sorted()
    for point in ordered.points:
        if higher_is_better and point.value >= target:
            return int(point.step)
        if not higher_is_better and point.value <= target:
            return int(point.step)
    return None


def load_baseline_value_from_results_table(
    results_table: Path,
    *,
    scenario: str,
    method: str,
    metric_column: str,
) -> float:
    with results_table.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("scenario") == scenario and row.get("method") == method:
                value = row.get(metric_column)
                if value is None:
                    raise KeyError(f"missing column {metric_column} in {results_table}")
                return float(value)
    raise KeyError(f"no row found for scenario={scenario!r}, method={method!r} in {results_table}")


def rate_series_per_1k_steps(series: ScalarSeries, *, smooth_window: int) -> ScalarSeries:
    ordered = series.sorted()
    if ordered.steps.size < 2:
        return ScalarSeries(name=f"{series.name}_rate", points=[])
    smoothed = smooth_values(ordered.values, smooth_window)
    rates = np.gradient(smoothed, ordered.steps) * 1000.0
    points = [ScalarPoint(step=int(step), value=float(rate)) for step, rate in zip(ordered.steps, rates)]
    return ScalarSeries(name=f"{series.name}_rate", points=points).sorted()


def _plot_series_panel(
    ax: plt.Axes,
    series_map: Mapping[str, ScalarSeries],
    *,
    title: str,
    ylabel: str,
    smooth_window: int,
    color: str,
) -> None:
    if not series_map:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training steps")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.grid(True, alpha=0.2)
        return

    run_names = list(series_map.keys())
    series_list = [series_map[name] for name in run_names]
    aggregated = aggregate_series(series_list)
    steps = aggregated["steps"]
    matrix = aggregated["matrix"]

    for run_name, series in series_map.items():
        ordered = series.sorted()
        values = smooth_values(ordered.values, smooth_window)
        steps_to_plot = ordered.steps
        ax.plot(steps_to_plot, values, linewidth=1.2, alpha=0.25, color=color)

    if steps.size > 0:
        mean_values = np.nanmean(matrix, axis=0)
        std_values = np.nanstd(matrix, axis=0)
        mean_values = smooth_values(np.nan_to_num(mean_values, nan=0.0), max(1, smooth_window))
        std_values = smooth_values(np.nan_to_num(std_values, nan=0.0), max(1, smooth_window))
        ax.plot(steps, mean_values, linewidth=2.5, color=color, label=f"{title} mean")
        ax.fill_between(steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.15)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Training steps")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)


def plot_experiment_1a(
    runs: TensorboardRunMap,
    output_path: Path,
    *,
    smooth_window: int = 5,
    title: str = "Experiment 1A: Training Convergence",
) -> Dict[str, Dict[str, float]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    summary: Dict[str, Dict[str, float]] = {}

    panel_specs = [
        ("episode_reward", "Episode reward", "Reward", "tab:blue"),
        ("validation_gar", "Validation GAR", "GAR", "tab:green"),
        ("validation_success_rate", "Validation success rate", "Success rate", "tab:orange"),
        ("validation_gar_delta_1k", "GAR improvement per 1k steps", "GAR / 1k steps", "tab:purple"),
        ("policy_entropy", "Policy entropy", "Entropy", "tab:red"),
        ("validation_latency_objective", "Validation latency objective", "Latency objective", "tab:brown"),
    ]

    for axis, (panel_key, panel_title, ylabel, color) in zip(axes_flat, panel_specs):
        series_map: Dict[str, ScalarSeries] = {}
        for run_name, tag_map in runs.items():
            if panel_key == "validation_gar_delta_1k":
                base = None
                for candidate in ("validation/gar", "episode/gar"):
                    if candidate in tag_map:
                        base = tag_map[candidate]
                        break
                if base is None:
                    continue
                series_map[run_name] = rate_series_per_1k_steps(base, smooth_window=smooth_window)
            else:
                panel_info = DEFAULT_1A_PANEL_LOOKUP.get(panel_key)
                if panel_info is None:
                    continue
                _, candidate_tags = panel_info
                for tag in candidate_tags:
                    if tag in tag_map:
                        series_map[run_name] = tag_map[tag]
                        break

        _plot_series_panel(
            axis,
            series_map,
            title=panel_title,
            ylabel=ylabel,
            smooth_window=smooth_window,
            color=color,
        )

        summary[panel_key] = {}
        for run_name, series in series_map.items():
            ordered = series.sorted()
            summary[panel_key][run_name] = {
                "final_step": float(ordered.points[-1].step) if ordered.points else 0.0,
                "final_value": float(ordered.points[-1].value) if ordered.points else 0.0,
                "max_value": float(np.nanmax(ordered.values)) if ordered.points else 0.0,
            }

    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return summary


def plot_experiment_1b(
    runs: TensorboardRunMap,
    output_path: Path,
    *,
    baseline_value: float,
    metric_tag: str = "validation/gar",
    smooth_window: int = 5,
    title: str = "Experiment 1B: Sample Efficiency",
) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series_by_run: Dict[str, ScalarSeries] = {}
    for run_name, tag_map in runs.items():
        if metric_tag in tag_map:
            series_by_run[run_name] = tag_map[metric_tag]
        elif metric_tag == "validation/gar" and "episode/gar" in tag_map:
            series_by_run[run_name] = tag_map["episode/gar"]

    if not series_by_run:
        raise ValueError(f"no runs contain metric tag {metric_tag!r}")

    aggregated = aggregate_series(list(series_by_run.values()))
    steps = aggregated["steps"]
    mean_values = smooth_values(aggregated["mean"], smooth_window)
    std_values = smooth_values(aggregated["std"], smooth_window)

    crossing_steps: Dict[str, Optional[int]] = {
        run_name: first_crossing_step(series, baseline_value, higher_is_better=True)
        for run_name, series in series_by_run.items()
    }
    valid_crossings = [step for step in crossing_steps.values() if step is not None]

    fig, (ax_curve, ax_hist) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for run_name, series in series_by_run.items():
        ordered = series.sorted()
        ax_curve.plot(ordered.steps, smooth_values(ordered.values, smooth_window), alpha=0.3, linewidth=1.2)

    ax_curve.plot(steps, mean_values, color="tab:blue", linewidth=2.5, label="Mean validation GAR")
    ax_curve.fill_between(steps, mean_values - std_values, mean_values + std_values, color="tab:blue", alpha=0.15)
    ax_curve.axhline(baseline_value, color="tab:red", linestyle="--", linewidth=2.0, label="Baseline target")
    if valid_crossings:
        parity_step = float(np.mean(valid_crossings))
        ax_curve.axvline(parity_step, color="tab:green", linestyle=":", linewidth=2.0, label="Mean parity step")
    ax_curve.set_title("Mean validation GAR vs training steps")
    ax_curve.set_xlabel("Training steps")
    ax_curve.set_ylabel("Validation GAR")
    ax_curve.grid(True, alpha=0.2)
    ax_curve.legend(loc="best", fontsize=8)

    hist_values = [step for step in crossing_steps.values() if step is not None]
    if hist_values:
        ax_hist.hist(hist_values, bins=min(8, len(hist_values)), color="tab:green", alpha=0.75, edgecolor="black")
        ax_hist.axvline(float(np.mean(hist_values)), color="black", linestyle="--", linewidth=1.8, label="Mean")
    else:
        ax_hist.text(0.5, 0.5, "No run reached target", transform=ax_hist.transAxes, ha="center", va="center")
    ax_hist.set_title("Steps to reach baseline target")
    ax_hist.set_xlabel("Training steps")
    ax_hist.set_ylabel("Run count")
    ax_hist.grid(True, alpha=0.2)
    if hist_values:
        ax_hist.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    summary = {
        "baseline_value": float(baseline_value),
        "metric_tag": metric_tag,
        "run_count": len(series_by_run),
        "crossing_steps": {run_name: (None if step is None else int(step)) for run_name, step in crossing_steps.items()},
        "mean_crossing_step": float(np.mean(valid_crossings)) if valid_crossings else None,
        "median_crossing_step": float(np.median(valid_crossings)) if valid_crossings else None,
        "final_mean": float(mean_values[-1]) if mean_values.size else None,
        "final_std": float(std_values[-1]) if std_values.size else None,
    }
    return summary


def write_json_summary(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _default_output_path(log_root: Path, stem: str) -> Path:
    return log_root / f"{stem}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report experiment helpers for convergence and sample efficiency")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("1a", help="Generate training convergence plots")
    p1.add_argument("--log-root", type=Path, required=True, help="Directory containing TensorBoard run folders")
    p1.add_argument("--output", type=Path, default=None, help="Output PNG path")
    p1.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary path")
    p1.add_argument("--smooth-window", type=int, default=5, help="Moving average window")

    p2 = subparsers.add_parser("1b", help="Generate sample-efficiency plots")
    p2.add_argument("--log-root", type=Path, required=True, help="Directory containing TensorBoard run folders")
    p2.add_argument("--output", type=Path, default=None, help="Output PNG path")
    p2.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary path")
    p2.add_argument("--baseline-value", type=float, default=None, help="Baseline GAR target")
    p2.add_argument("--results-table", type=Path, default=None, help="CSV results table for auto baseline lookup")
    p2.add_argument("--scenario", type=str, default=None, help="Scenario name in the results table")
    p2.add_argument("--baseline-method", type=str, default="best_fit", help="Baseline method name in the results table")
    p2.add_argument("--metric-column", type=str, default="gar_capacity_mean", help="Metric column for the baseline target")
    p2.add_argument("--metric-tag", type=str, default="validation/gar", help="TensorBoard tag to analyze")
    p2.add_argument("--smooth-window", type=int, default=5, help="Moving average window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_tensorboard_runs(args.log_root)

    if args.command == "1a":
        output = args.output or _default_output_path(args.log_root, "experiment_1a_training_convergence")
        summary = plot_experiment_1a(runs, output, smooth_window=max(1, int(args.smooth_window)))
        summary_path = args.summary_json or output.with_suffix(".json")
        write_json_summary(summary_path, summary)
        print(f"Wrote {output}")
        print(f"Wrote {summary_path}")
        return

    baseline_value = args.baseline_value
    if baseline_value is None:
        if args.results_table is None or args.scenario is None:
            raise ValueError("provide either --baseline-value or both --results-table and --scenario")
        baseline_value = load_baseline_value_from_results_table(
            args.results_table,
            scenario=args.scenario,
            method=args.baseline_method,
            metric_column=args.metric_column,
        )

    output = args.output or _default_output_path(args.log_root, "experiment_1b_sample_efficiency")
    summary = plot_experiment_1b(
        runs,
        output,
        baseline_value=float(baseline_value),
        metric_tag=args.metric_tag,
        smooth_window=max(1, int(args.smooth_window)),
    )
    summary_path = args.summary_json or output.with_suffix(".json")
    write_json_summary(summary_path, summary)
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()