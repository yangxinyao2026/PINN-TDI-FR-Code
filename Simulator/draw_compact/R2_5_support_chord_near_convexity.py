"""Compare boundary sampling density in four equal-size system panels.

Export four identically sized PDFs, near_convexity_2x2.pdf, and legend.pdf.
Reading order: 33-bus, 118-bus, 533-bus, and 36-bus three-phase systems.
No legend, titles, cross markers, or zoom panels appear inside the data figures.
The separate legend has two vertically stacked line-and-marker entries.
Legend wording follows the requested manuscript descriptions. The second series
still comes from AC projections of hull-edge samples; this plotting script does
not compute visibility from a reference point.

Retain the original selection: the condition maximizing projection distance /
support-hull diameter over all successful chord tests in each system. Show all
360 AC support points as hollow circles and all 3240 projected hull-edge samples
as small translucent dots. A thin black support hull guides the eye. Projection
samples are neither joined nor augmented with support endpoints. No thinning,
jitter, smoothing, or optimization is applied.

All plotting boxes have the same 4:3 physical aspect and equal P/Q scales.
Fixed ticks include both ends of every axis, aligning all four frame edges
with tick positions. The displayed ranges retain every sample without stretching.
Standalone pages use a shared content crop with 1.5 mm padding so their final
dimensions match. Fonts and axes follow the existing manuscript style.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
from matplotlib.transforms import Bbox
import numpy as np
from scipy.spatial import ConvexHull


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
               / "support_chord_convexity")
OUTPUT_ROOT = PROJECT_ROOT / "results" / "pictures_compact" / "support_chord_near_convexity"
CASES = (
    ("case33bw_ds", "33-bus system"),
    ("case118zh_ds", "118-bus system"),
    ("case533mt_hi_ds", "533-bus system"),
    ("case36real_3phase_ds", "36-bus three-phase system"),
)
LABELS = ("Support-point convex hull", "AC-feasible projected samples", "AC support points")
LEGEND_LABELS = (
    "Support points joined into a polygon (without reference)",
    "True region visible from the reference point",
)
PROJECTION_COLOR = "#D98C4A"
SUPPORT_COLOR = "#4C72B0"
PROJECTION_SIZE = 3.5
SUPPORT_SIZE = 8.0
PANEL_WIDTH_MM, PANEL_HEIGHT_MM = 89.0, 68.0
LEFT_MM, RIGHT_MM, BOTTOM_MM = 16.5, 3.0, 13.0
AXES_WIDTH_MM = PANEL_WIDTH_MM - LEFT_MM - RIGHT_MM
AXES_ASPECT = 4.0 / 3.0
AXES_HEIGHT_MM = AXES_WIDTH_MM / AXES_ASPECT
EXPORT_PADDING_MM = 1.5
# Each viewport has a 4:3 range ratio, matching the common plotting box.
AXIS_TICKS = {
    "case33bw_ds": (
        np.array([0.26, 0.32, 0.38, 0.44, 0.50]),
        np.array([0.14, 0.20, 0.26, 0.32]),
    ),
    "case118zh_ds": (
        np.array([1.4, 1.9, 2.4, 2.9, 3.4]),
        np.array([1.1, 1.6, 2.1, 2.6]),
    ),
    "case533mt_hi_ds": (
        np.array([0.5, 0.7, 0.9, 1.1, 1.3]),
        np.array([-0.30, -0.15, 0.0, 0.15, 0.30]),
    ),
    "case36real_3phase_ds": (
        np.array([0.6, 0.9, 1.2, 1.5, 1.8]),
        np.array([0.0, 0.3, 0.6, 0.9]),
    ),
}


def point(row: dict, prefix: str) -> np.ndarray:
    return np.array([float(row[f"{prefix}_p"]), float(row[f"{prefix}_q"])])


def is_valid_projection(row: dict) -> bool:
    if row["projection_success"].strip().lower() != "true":
        return False
    try:
        values = [float(row[key]) for key in ("distance_pu", "projection_p", "projection_q")]
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(values).all())


def projected_samples(rows: list[dict], points: np.ndarray,
                      hull_indices: np.ndarray) -> np.ndarray:
    """Validate edge provenance and return each measured projection once."""
    by_edge = defaultdict(list)
    for row in rows:
        edge = (int(row["endpoint_index_1"]), int(row["endpoint_index_2"]))
        by_edge[edge].append(row)
    edges = list(zip(hull_indices, np.roll(hull_indices, -1)))
    if set(by_edge) != set(edges):
        raise ValueError("Cached edge connectivity does not match the support hull.")
    for first, second in edges:
        samples = sorted(by_edge[(first, second)], key=lambda row: float(row["lambda"]))
        np.testing.assert_allclose([float(row["lambda"]) for row in samples],
                                   np.arange(1, 10) / 10, rtol=0, atol=1e-14)
        for row in samples:
            if not is_valid_projection(row):
                raise ValueError("Incomplete hull-edge projections.")
            np.testing.assert_allclose(point(row, "endpoint_1"), points[first], rtol=0, atol=1e-12)
            np.testing.assert_allclose(point(row, "endpoint_2"), points[second], rtol=0, atol=1e-12)
            lam = float(row["lambda"])
            target = (1 - lam) * points[first] + lam * points[second]
            np.testing.assert_allclose(point(row, "target"), target, rtol=0, atol=1e-12)
    return np.asarray([point(row, "projection") for row in rows])


def load_case_result(case_name: str) -> dict:
    folder = RESULT_ROOT / case_name
    with np.load(folder / "support_data.npz", allow_pickle=False) as saved:
        points = np.asarray(saved["support_points"], dtype=float)
        success = np.asarray(saved["support_success"], dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 2 or success.shape != points.shape[:2]:
        raise ValueError(f"Unexpected support data shape: {case_name}")
    geometries, diameters = [], []
    for condition_points, ok in zip(points, success):
        indices = np.flatnonzero(ok & np.isfinite(condition_points).all(axis=1))
        hull_indices = indices[ConvexHull(condition_points[indices]).vertices]
        vertices = condition_points[hull_indices]
        diameter = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2).max()
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError(f"Degenerate support hull: {case_name}")
        geometries.append((vertices, hull_indices))
        diameters.append(diameter)
    with (folder / "line_tests.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    successful = [row for row in rows if is_valid_projection(row)]
    if not successful:
        raise ValueError(f"No successful projections: {case_name}")
    worst = max(successful, key=lambda row: float(row["distance_pu"])
                / diameters[int(row["condition_index"])])
    condition = int(worst["condition_index"])
    vertices, hull_indices = geometries[condition]
    edge_rows = [row for row in rows if int(row["condition_index"]) == condition
                 and row["line_type"] == "hull_edge"]
    projections = projected_samples(edge_rows, points[condition], hull_indices)
    valid_support = success[condition] & np.isfinite(points[condition]).all(axis=1)
    return {
        "case": case_name,
        "condition": condition,
        "hull": np.vstack((vertices, vertices[0])),
        "support": points[condition][valid_support],
        "projected": projections,
        "projection_count": len(edge_rows),
        "max_relative_distance": float(worst["distance_pu"]) / diameters[condition],
    }


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.unicode_minus": False,
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
        "path.simplify": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def draw_panel(ax, result: dict) -> None:
    all_points = np.vstack((result["support"], result["projected"]))
    x_ticks, y_ticks = AXIS_TICKS[result["case"]]
    lower = np.array([x_ticks[0], y_ticks[0]])
    upper = np.array([x_ticks[-1], y_ticks[-1]])
    if not (np.all(all_points > lower) and np.all(all_points < upper)):
        raise ValueError(f"Samples exceed the fixed axis limits: {result['case']}")
    np.testing.assert_allclose((upper[0] - lower[0]) / (upper[1] - lower[1]),
                               AXES_ASPECT, rtol=1e-12)
    hull, projected, support = result["hull"], result["projected"], result["support"]
    ax.plot(hull[:, 0], hull[:, 1], color="black", linewidth=0.8,
            linestyle="-", marker="None", label=LABELS[0], zorder=2)
    ax.scatter(projected[:, 0], projected[:, 1], s=PROJECTION_SIZE,
               color=PROJECTION_COLOR, alpha=0.45, linewidths=0,
               label=LABELS[1], zorder=3)
    ax.scatter(support[:, 0], support[:, 1], s=SUPPORT_SIZE,
               facecolors="none", edgecolors=SUPPORT_COLOR, linewidths=0.55,
               label=LABELS[2], zorder=4)
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
           xlabel="Active power at TDI (p.u.)", ylabel="Reactive power at TDI (p.u.)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color="#D9D9D9", linestyle="--", linewidth=0.5, alpha=0.75)
    for axis, ticks in ((ax.xaxis, x_ticks), (ax.yaxis, y_ticks)):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.tick_params(direction="out")


def add_panel(fig, result: dict, origin_mm=(0.0, 0.0)):
    width_mm, height_mm = fig.get_size_inches() * 25.4
    ax = fig.add_axes(((origin_mm[0] + LEFT_MM) / width_mm,
                       (origin_mm[1] + BOTTOM_MM) / height_mm,
                       AXES_WIDTH_MM / width_mm, AXES_HEIGHT_MM / height_mm))
    draw_panel(ax, result)
    return ax


def make_figure(result: dict):
    fig = plt.figure(figsize=(PANEL_WIDTH_MM / 25.4, PANEL_HEIGHT_MM / 25.4))
    add_panel(fig, result)
    return fig


def make_combined_figure(results: list[dict]):
    gap_x_mm, gap_y_mm = 5.0, 6.0
    width_mm = 2 * PANEL_WIDTH_MM + gap_x_mm
    height_mm = 2 * PANEL_HEIGHT_MM + gap_y_mm
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    for index, result in enumerate(results):
        row, column = divmod(index, 2)
        origin = (column * (PANEL_WIDTH_MM + gap_x_mm),
                  (1 - row) * (PANEL_HEIGHT_MM + gap_y_mm))
        add_panel(fig, result, origin)
    return fig


def make_legend():
    fig = plt.figure(figsize=(183 / 25.4, 18 / 25.4))
    handles = [
        Line2D([], [], color="black", linewidth=0.8, linestyle="-",
               marker="o", markersize=np.sqrt(SUPPORT_SIZE),
               markerfacecolor="none", markeredgecolor=SUPPORT_COLOR, markeredgewidth=0.55),
        Line2D([], [], color="black", linewidth=0.8, linestyle="-",
               marker="o", markersize=np.sqrt(PROJECTION_SIZE),
               markerfacecolor=to_rgba(PROJECTION_COLOR, 0.45), markeredgewidth=0),
    ]
    fig.legend(handles, LEGEND_LABELS, loc="center", ncol=1,
               frameon=False, fontsize=9, numpoints=1, markerscale=1.5,
               handlelength=2.5, labelspacing=0.55, borderaxespad=0, borderpad=0)
    return fig


def save_pdf(fig, path: Path, title: str, subject: str = "", crop="tight") -> None:
    with BytesIO() as buffer:
        fig.savefig(buffer, format="pdf", bbox_inches=crop,
                    pad_inches=EXPORT_PADDING_MM / 25.4,
                    metadata={"Title": title, "Subject": subject})
        path.write_bytes(buffer.getvalue())
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    set_style()
    results = [(load_case_result(case), display_name) for case, display_name in CASES]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figures = [make_figure(result) for result, _ in results]
    boxes = []
    for fig in figures:
        fig.canvas.draw()
        boxes.append(fig.get_tightbbox(fig.canvas.get_renderer()))
    common_crop = Bbox.union(boxes).padded(EXPORT_PADDING_MM / 25.4)
    for fig, (result, display_name) in zip(figures, results):
        path = OUTPUT_ROOT / f"{result['case']}.pdf"
        save_pdf(fig, path, f"{display_name}: support and projected boundary samples",
                 f"Condition {result['condition'] + 1}; {len(result['support'])} support points; "
                 f"{result['projection_count']} projected edge samples", common_crop)
    combined = make_combined_figure([result for result, _ in results])
    save_pdf(combined, OUTPUT_ROOT / "near_convexity_2x2.pdf",
             "Support and projected boundary samples in four systems",
             "Top: 33-bus, 118-bus. Bottom: 533-bus, 36-bus three-phase.")
    save_pdf(make_legend(), OUTPUT_ROOT / "legend.pdf", "Support/chord comparison legend")


if __name__ == "__main__":
    main()
