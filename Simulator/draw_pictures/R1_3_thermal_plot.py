# -*- coding: utf-8 -*-
"""Plot the formal R1.3 comparison with and without branch thermal limits."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from Simulator import PROJECT_ROOT


ROOT = PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison" / "thermal" / "case33bw_ds"
BASELINE = ROOT / "baseline" / "evaluation_data.npz"
DEFAULT_MARGIN = 1.20


def thermal_dir(margin: float) -> Path:
    return ROOT / f"thermal_{int(round(100.0 * margin)):03d}pct"


def ordered_points(points):
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return points
    centre = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return points[np.argsort(angles)]


def close_curve(points):
    points = np.asarray(points, dtype=float)
    return points if len(points) == 0 else np.vstack((points, points[0]))


def polygon_vertices(A, b, tol=1e-7):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    points = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            matrix = np.vstack((A[i], A[j]))
            if abs(np.linalg.det(matrix)) <= 1e-12:
                continue
            point = np.linalg.solve(matrix, np.array([b[i], b[j]]))
            if np.all(A @ point <= b + tol):
                points.append(point)
    if not points:
        return np.empty((0, 2), dtype=float)
    return ordered_points(np.unique(np.round(np.asarray(points), decimals=10), axis=0))


def finite(data):
    data = np.asarray(data, dtype=float).reshape(-1)
    return data[np.isfinite(data)]


def _array(data, *keys):
    for key in keys:
        if key in data.files:
            return np.asarray(data[key])
    raise KeyError(f"Evaluation file is missing fields: {keys}")


def main(margin=DEFAULT_MARGIN):
    margin = float(margin)
    thermal_root = thermal_dir(margin)
    thermal_path = thermal_root / "evaluation_data.npz"
    missing = [path for path in (BASELINE, thermal_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run the formal baseline and thermal evaluations first:\n"
            + "\n".join(str(path) for path in missing)
        )

    baseline = np.load(BASELINE, allow_pickle=True)
    thermal = np.load(thermal_path, allow_pickle=True)
    baseline_color = "#2878B5"
    thermal_color = "#C82423"
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    for data, color, label in (
        (baseline, baseline_color, "Without thermal limits"),
        (thermal, thermal_color, f"With thermal limits ($\\kappa$={margin:.2f})"),
    ):
        true_points = _array(data, "true_support_points")[0]
        true_points = ordered_points(true_points[np.all(np.isfinite(true_points), axis=1)])
        if len(true_points):
            curve = close_curve(true_points)
            ax.plot(curve[:, 0], curve[:, 1], color=color, lw=2.0, label=f"{label}: true")
        vertices = polygon_vertices(_array(data, "predicted_A")[0], _array(data, "predicted_b")[0])
        if len(vertices):
            curve = close_curve(vertices)
            ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.6, ls="--", label=f"{label}: PINN")
    ax.set_xlabel("Active power at TDI (p.u.)")
    ax.set_ylabel("Reactive power at TDI (p.u.)")
    ax.set_title("(a) Nominal flexibility regions")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    coverage_base = finite(100.0 * np.nanmean(_array(baseline, "coverage"), axis=1))
    coverage_thermal = finite(100.0 * np.nanmean(_array(thermal, "coverage"), axis=1))
    box = ax.boxplot(
        [coverage_base, coverage_thermal],
        tick_labels=["No thermal\nlimits", f"Thermal\n$\\kappa$={margin:.2f}"],
        patch_artist=True,
        showmeans=True,
    )
    for patch, color in zip(box["boxes"], (baseline_color, thermal_color)):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.set_ylabel("Coverage ratio (%)")
    ax.set_title("(b) Coverage across operating conditions")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    metric_specs = (("Feasibility", ("feasibility",)), ("Optimality", ("optimality",)))
    x = np.arange(len(metric_specs), dtype=float)
    width = 0.34
    for offset, data, color, label in (
        (-width / 2, baseline, baseline_color, "No thermal limits"),
        (width / 2, thermal, thermal_color, f"Thermal $\\kappa$={margin:.2f}"),
    ):
        means, p95s = [], []
        for _, keys in metric_specs:
            values = finite(_array(data, *keys))
            means.append(float(np.mean(values)))
            p95s.append(float(np.percentile(values, 95)))
        ax.bar(x + offset, means, width=width, color=color, alpha=0.72, label=label)
        ax.scatter(x + offset, p95s, color="black", marker="_", s=90, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([item[0] for item in metric_specs])
    ax.set_yscale("log")
    ax.set_ylabel("Squared Euclidean distance")
    ax.set_title("(c) Approximation errors (bar: mean; mark: P95)")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    suffix = int(round(100 * margin))
    output_png = thermal_root / f"thermal_comparison_{suffix:03d}pct.png"
    output_pdf = thermal_root / f"thermal_comparison_{suffix:03d}pct.pdf"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Formal comparison figure: {output_png}")
    return output_png, output_pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    args = parser.parse_args()
    main(args.margin)
