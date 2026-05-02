from __future__ import annotations

import tempfile
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.report_experiments import (
    ScalarPoint,
    ScalarSeries,
    build_series,
    first_crossing_step,
    load_baseline_value_from_results_table,
    plot_experiment_1a,
    plot_experiment_1b,
    smooth_values,
)


def _make_series(name: str, pairs: list[tuple[int, float]]) -> ScalarSeries:
    return ScalarSeries(name=name, points=[ScalarPoint(step=step, value=value) for step, value in pairs])


class ReportExperimentSmokeTests(unittest.TestCase):
    def test_smoothing_and_threshold(self) -> None:
        series = build_series([(0, 0.10), (100, 0.20), (200, 0.60), (300, 0.70)], name="validation/gar")
        smoothed = smooth_values(series.values, 3)
        self.assertEqual(len(smoothed), 4)
        self.assertAlmostEqual(first_crossing_step(series, 0.5), 200)

    def test_load_baseline_value_from_results_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            table_path = Path(tmp_dir) / "results.csv"
            table_path.write_text(
                "scenario,method,gar_capacity_mean\n"
                "high_load,rl_full,0.67\n"
                "high_load,best_fit,0.68\n"
            )
            value = load_baseline_value_from_results_table(
                table_path,
                scenario="high_load",
                method="best_fit",
                metric_column="gar_capacity_mean",
            )
            self.assertAlmostEqual(value, 0.68)

    def test_plot_experiment_1a_creates_png(self) -> None:
        runs = {
            "run_a": {
                "rollout/ep_rew_mean": _make_series("rollout/ep_rew_mean", [(0, 0.0), (100, 0.2), (200, 0.5)]),
                "validation/gar": _make_series("validation/gar", [(0, 0.20), (100, 0.35), (200, 0.55)]),
                "validation/success_rate": _make_series("validation/success_rate", [(0, 0.4), (100, 0.45), (200, 0.5)]),
                "train/entropy_loss": _make_series("train/entropy_loss", [(0, -1.0), (100, -0.8), (200, -0.7)]),
                "validation/latency_objective": _make_series("validation/latency_objective", [(0, 2.0), (100, 1.8), (200, 1.5)]),
            },
            "run_b": {
                "rollout/ep_rew_mean": _make_series("rollout/ep_rew_mean", [(0, 0.05), (100, 0.25), (200, 0.6)]),
                "validation/gar": _make_series("validation/gar", [(0, 0.18), (100, 0.30), (200, 0.52)]),
                "validation/success_rate": _make_series("validation/success_rate", [(0, 0.35), (100, 0.43), (200, 0.49)]),
                "train/entropy_loss": _make_series("train/entropy_loss", [(0, -1.2), (100, -0.9), (200, -0.6)]),
                "validation/latency_objective": _make_series("validation/latency_objective", [(0, 2.2), (100, 1.9), (200, 1.6)]),
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "exp1a.png"
            summary = plot_experiment_1a(runs, output_path, smooth_window=2)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertIn("validation_gar", summary)
            self.assertIn("run_a", summary["validation_gar"])

    def test_plot_experiment_1b_creates_png_and_summary(self) -> None:
        runs = {
            "seed_1": {
                "validation/gar": _make_series("validation/gar", [(0, 0.20), (100, 0.32), (200, 0.50)]),
            },
            "seed_2": {
                "validation/gar": _make_series("validation/gar", [(0, 0.18), (100, 0.34), (200, 0.58)]),
            },
            "seed_3": {
                "validation/gar": _make_series("validation/gar", [(0, 0.22), (100, 0.31), (200, 0.47)]),
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "exp1b.png"
            summary = plot_experiment_1b(runs, output_path, baseline_value=0.45, smooth_window=2)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(summary["run_count"], 3)
            self.assertIn("crossing_steps", summary)


if __name__ == "__main__":
    unittest.main()