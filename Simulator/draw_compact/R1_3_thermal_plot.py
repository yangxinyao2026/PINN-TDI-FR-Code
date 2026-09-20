"""Export only the nominal-region panel of the formal thermal-limit comparison.

Compare reference and PINN regions with/without branch thermal limits at the same
nominal operating condition. Reuse the original support-point ordering and
half-space intersection method, without recomputing or smoothing boundaries.
Solid/dashed lines denote reference/PINN; blue/red denote without/with thermal limits.
Use a 120 x 110 mm canvas; trim PDF to visible content with a 1 mm outer pad.
All text uses 9 pt Times New Roman.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np


SHARE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    SHARE_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "thermal" / "case33bw_ds"
)
OUTPUT_DIR = SHARE_ROOT / "results" / "pictures_compact" / "thermal_comparison"
DEFAULT_MARGIN = 1.20
FONT_SIZE = 9
LABELS = (
    "Without thermal limits: reference",
    "Without thermal limits: PINN",
    "With thermal limits: reference",
    "With thermal limits: PINN",
)


def ordered_points(points):
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return points
    centre = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return points[np.argsort(angles)]


def close_curve(points):
    return np.vstack((points, points[0]))


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
        raise ValueError("The predicted half-spaces do not produce polygon vertices.")
    return ordered_points(np.unique(np.round(points, decimals=10), axis=0))


def load_regions(data_root, margin):
    suffix = int(round(100 * margin))
    regions = []
    reference_condition = reference_directions = None
    for folder in ("baseline", f"thermal_{suffix:03d}pct"):
        path = data_root / folder / "evaluation_data.npz"
        with np.load(path, allow_pickle=False) as data:
            condition = data["test_dthetas"][0]
            directions = data["eval_dirs"]
            if not np.allclose(condition, 0, rtol=0, atol=1e-12):
                raise ValueError(f"The first condition is not nominal: {path}")
            if reference_condition is None:
                reference_condition = condition.copy()
                reference_directions = directions.copy()
            elif not (
                np.array_equal(condition, reference_condition)
                and np.array_equal(directions, reference_directions)
            ):
                raise ValueError("The nominal conditions/directions do not match.")
            if folder != "baseline" and not np.isclose(float(data["current_margin"]), margin):
                raise ValueError("Stored thermal margin differs from the requested margin.")
            true_points = np.asarray(data["true_support_points"][0], dtype=float)
            if not np.isfinite(true_points).all():
                raise ValueError(f"Incomplete nominal true-region boundary: {path}")
            A, b = data["predicted_A"][0], data["predicted_b"][0]
            if not (np.isfinite(A).all() and np.isfinite(b).all()):
                raise ValueError(f"Nonfinite predicted half-spaces: {path}")
            true_boundary = ordered_points(true_points)
            predicted_boundary = polygon_vertices(A, b)
            if min(len(true_boundary), len(predicted_boundary)) < 3:
                raise ValueError("A region has fewer than three boundary points.")
            regions.append((true_boundary, predicted_boundary))
            print(f"{folder}: true support points={len(true_boundary)}, PINN vertices={len(predicted_boundary)}")
    return regions


def configure_style():
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.linewidth": 0.65,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.fallback": None,
        "pdf.fonttype": 42,
        "pdf.compression": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def make_figure(regions):
    fig, ax = plt.subplots(figsize=(120 / 25.4, 110 / 25.4))
    fig.subplots_adjust(left=0.16, right=0.975, bottom=0.14, top=0.975)
    for index, ((true_boundary, predicted_boundary), color) in enumerate(
        zip(regions, ("#2878B5", "#C82423"))
    ):
        for curve_index, (points, style) in enumerate(
            ((true_boundary, "-"), (predicted_boundary, "--"))
        ):
            curve = close_curve(points)
            ax.plot(curve[:, 0], curve[:, 1], color=color, linestyle=style,
                    linewidth=1.4, label=LABELS[index * 2 + curve_index])
    ax.set_xlabel("Active power at TDI (p.u.)", labelpad=5)
    ax.set_ylabel("Reactive power at TDI (p.u.)", labelpad=5)
    ax.set_aspect("equal", adjustable="box")
    all_points = np.vstack([boundary for pair in regions for boundary in pair])
    for dimension, set_ticks, set_limits in (
        (0, ax.set_xticks, ax.set_xlim),
    ):
        locator = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        ticks = np.round(locator.tick_values(
            all_points[:, dimension].min(), all_points[:, dimension].max()
        ), decimals=12)
        set_ticks(ticks)
        set_limits(ticks[0], ticks[-1])
    ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35])
    ax.set_ylim(0.15, 0.35)
    ax.set_axisbelow(True)
    ax.grid(False, which="both")
    ax.tick_params(axis="both", which="major", length=3, pad=4)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
              ncol=2, frameon=False, fontsize=FONT_SIZE, handlelength=1.6,
              handletextpad=0.5, columnspacing=0.9, labelspacing=0.65,
              borderaxespad=0.3)
    return fig


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not np.isfinite(args.margin) or args.margin <= 0:
        parser.error("--margin must be a positive finite number.")
    regions = load_regions(args.data_root, args.margin)
    configure_style()
    fig = make_figure(regions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"thermal_regions_{int(round(100 * args.margin)):03d}pct.pdf"
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=1 / 25.4, metadata={
        "Title": "Effect of branch thermal limits on nominal flexibility regions",
        "Subject": f"case33bw_ds; current margin={args.margin:.2f}; nominal condition",
        "Creator": "Matplotlib",
    })
    plt.close(fig)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
