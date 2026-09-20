"""Export separate region PDFs for the two nonconvex experiments.

Sources: runners/main_revise2_5_two_disks.py and main_revise2_5_star.py.
Reuse their coverage_raw.npz caches: polygon radius = rho * true radius.
The 120 evaluated polygon points are unchanged; no training is performed.
Projection probing is labelled "without reference", radial probing "with reference".
The reference point is (0, 0) in both experiments.

Two disks: centres (+/-0.8, 0), radius 1. Star: R=1, eps=0.30, k=4.
True outlines use the same sampling as the original panels (120 and 400 points).
The disk union receives one uniform fill, avoiding a darker overlap artefact.
Equal coordinate scales preserve the geometry and permit comparison across panels.
Export two_disks.pdf and star.pdf without legends, plus a separate legend.pdf.
The legend has two rows: True region / Reference point in the left column,
and the approximations without / with reference in the right column.
All PDFs are cropped to their contents with a 1.5 mm outer margin.
Both figures omit grids and use y limits [-1.5, 1.5]; x limits are [-2, 2]
for two disks and [-1.5, 1.5] for the star.
"""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "results" / "ds_proj_revise_V1"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "pictures_compact" / "nonconvex_comparison"
EXPORT_PADDING_MM = 1.5
METHODS = ("projection", "radial")
LABELS = (
    "True region",
    "Polygonal approximation (without reference)",
    "Polygonal approximation (with reference)",
    "Reference point",
)
TRUE_COLOR = "#C44E52"
TRUE_FILL = to_rgba("lightcoral", 0.24)
METHOD_COLORS = {"projection": "#4C72B0", "radial": "#55A868"}
METHOD_STYLES = {"projection": (0, (4, 2)), "radial": "-"}
X_LIMITS = {"two_disks": (-2.0, 2.0), "star": (-1.5, 1.5)}
Y_LIMITS = (-1.5, 1.5)


def true_radius(case: str, directions: np.ndarray) -> np.ndarray:
    """Original analytic radial functions, simplified for reference (0, 0)."""
    if case == "two_disks":
        d, radius = 0.8, 1.0
        return d * np.abs(directions[:, 0]) + np.sqrt(
            radius**2 - d**2 * directions[:, 1] ** 2
        )
    if case == "star":
        theta = np.arctan2(directions[:, 1], directions[:, 0])
        return 1.0 * (1.0 + 0.30 * np.cos(4 * theta))
    raise ValueError(f"Unknown case: {case}")


def load_panel(case: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    folder = DATA_ROOT / f"synthetic_{case}"
    with np.load(folder / "coverage_raw.npz", allow_pickle=False) as raw:
        methods = raw["methods"].tolist()
        directions = raw["directions"].copy()
        rho = raw["rho"].copy()
    if methods != list(METHODS) or rho.shape != (2, 120):
        raise ValueError(f"Unexpected cached methods or coverage shape: {folder}")
    if not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError(f"Invalid cached radial coverage: {folder}")
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    np.testing.assert_allclose(
        directions, np.column_stack((np.cos(theta), np.sin(theta))), atol=1e-14
    )
    with (folder / "comparison_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        summaries = {row["method"]: row for row in csv.DictReader(stream)}
    polygons = {}
    radius = true_radius(case, directions)
    for method, coverage in zip(methods, rho):
        summary = summaries[method]
        np.testing.assert_allclose(
            [coverage.mean(), np.median(coverage), coverage.min(), coverage.max()],
            [float(summary[key]) for key in ("rho_mean", "rho_median", "rho_min", "rho_max")],
            rtol=1e-12,
        )
        polygons[method] = (coverage * radius)[:, None] * directions
    if case == "star":
        theta = np.linspace(0, 2 * np.pi, 400)
        directions = np.column_stack((np.cos(theta), np.sin(theta)))
        radius = true_radius(case, directions)
    boundary = radius[:, None] * directions
    for points in (boundary, *polygons.values()):
        if not (np.all(points[:, 0] > X_LIMITS[case][0])
                and np.all(points[:, 0] < X_LIMITS[case][1])
                and np.all(points[:, 1] > Y_LIMITS[0])
                and np.all(points[:, 1] < Y_LIMITS[1])):
            raise ValueError(f"Geometry exceeds the plotting limits: {case}")
    return boundary, polygons


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def draw_panel(ax, case: str, boundary: np.ndarray, polygons: dict) -> None:
    ax.fill(boundary[:, 0], boundary[:, 1], facecolor=TRUE_FILL,
            edgecolor="none", zorder=1)
    closed = np.vstack((boundary, boundary[0]))
    ax.plot(closed[:, 0], closed[:, 1], color=TRUE_COLOR, linewidth=1.4, zorder=3)
    for method in METHODS:
        closed = np.vstack((polygons[method], polygons[method][0]))
        ax.plot(closed[:, 0], closed[:, 1], color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method], linewidth=1.3, zorder=4)
    ax.plot(0, 0, marker="*", markersize=8, color="black", linestyle="none", zorder=5)
    ax.set(xlim=X_LIMITS[case], ylim=Y_LIMITS, xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(False, which="both")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.tick_params(direction="out")


def main() -> None:
    panels = [(case, *load_panel(case)) for case in ("two_disks", "star")]
    set_style()
    handles = [
        Patch(facecolor=TRUE_FILL, edgecolor=TRUE_COLOR, linewidth=1.2),
        *(Line2D([], [], color=METHOD_COLORS[method], linestyle=METHOD_STYLES[method],
                 linewidth=1.3) for method in METHODS),
        Line2D([], [], marker="*", markersize=8, color="black", linestyle="none"),
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for case, boundary, polygons in panels:
        fig, ax = plt.subplots(figsize=(105 / 25.4, 78 / 25.4))
        fig.subplots_adjust(left=0.14, right=0.975, bottom=0.17, top=0.96)
        draw_panel(ax, case, boundary, polygons)
        output_path = OUTPUT_ROOT / f"{case}.pdf"
        with BytesIO() as buffer:
            fig.savefig(buffer, format="pdf", bbox_inches="tight",
                        pad_inches=EXPORT_PADDING_MM / 25.4, metadata={
                "Title": f"{case.replace('_', ' ').capitalize()}: region and polygonal approximations",
                "Subject": "Cached projection and radial results; reference at the origin",
            })
            output_path.write_bytes(buffer.getvalue())
        plt.close(fig)
        print(f"Saved: {output_path}")

    # Matplotlib fills legend columns first.
    legend_order = (0, 3, 1, 2)
    legend_fig = plt.figure(figsize=(135 / 25.4, 16 / 25.4))
    legend_fig.legend([handles[i] for i in legend_order],
                      [LABELS[i] for i in legend_order],
                      loc="center", ncol=2, frameon=False, fontsize=9,
                      handlelength=2.5, columnspacing=1.8,
                      labelspacing=0.55, borderaxespad=0, borderpad=0)
    legend_path = OUTPUT_ROOT / "legend.pdf"
    with BytesIO() as buffer:
        legend_fig.savefig(buffer, format="pdf", bbox_inches="tight",
                           pad_inches=EXPORT_PADDING_MM / 25.4,
                           metadata={"Title": "Nonconvex region comparison legend"})
        legend_path.write_bytes(buffer.getvalue())
    plt.close(legend_fig)
    print(f"Saved: {legend_path}")


if __name__ == "__main__":
    main()
