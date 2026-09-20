# -*- coding: utf-8 -*-
"""Plot R1.1 learned polygons against a 360-support-point AC hull.

The script uses the finalized R1.1 polygon bank and its common test conditions.
For the fixed 10th common condition, it obtains 360 AC support points by
direction-wise optimization and caches them.  The displayed hull is a
finite-direction support-point hull, not a mathematical exact feasible-domain
boundary.
"""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from Simulator import PROJECT_ROOT
from Simulator.runners.main_revise2_5_support_chord_convexity import (
    directions_even,
    load_case,
    update_parameters,
)


R11_ROOT = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "learning_methods" / "case33bw_ds"
    / "lin_paper_2d_formal_validated_timed"
)
POLYGON_BANK = R11_ROOT / "models" / "independent_polygon_bank.npz"
TRUTH_CACHE = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "supervised" / "case33bw_ds" / "support_points" / "evaluation"
    / "test_truth.npz"
)
UPDATED_EVALUATION = R11_ROOT / "evaluation" / "lin_paper_updated_raw.npz"
OUTPUT_DIR = R11_ROOT / "figures"
N_SUPPORT_DIRECTIONS = 360
SELECTED_SCENARIO_ONE_BASED = 10


def halfspace_vertices(A, b, tolerance=1e-8):
    """Return counter-clockwise vertices of a bounded 2-D polytope Ax<=b."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    candidates = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            matrix = np.stack((A[i], A[j]))
            if abs(np.linalg.det(matrix)) <= 1e-12:
                continue
            point = np.linalg.solve(matrix, np.array((b[i], b[j])))
            if np.all(A @ point <= b + tolerance):
                candidates.append(point)
    if len(candidates) < 3:
        raise RuntimeError("The learned halfspaces do not form a bounded polygon.")
    candidates = np.unique(np.round(np.asarray(candidates), 12), axis=0)
    return candidates[ConvexHull(candidates).vertices]


def closed(points):
    return np.vstack((points, points[0]))


def coverage_text(rho):
    under = np.maximum(1.0 - rho, 0.0).mean()
    over = np.maximum(rho - 1.0, 0.0).mean()
    return (
        f"mean rho = {rho.mean():.4f}\n"
        f"mean under = {100 * under:.3f}%\n"
        f"mean over = {100 * over:.3f}%"
    )


def load_or_compute_support_points(scenario, condition):
    """Return 360 true AC support points, reusing only an exact cache match."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / f"scenario_{scenario:03d}_support_360.npz"
    angles, directions = directions_even(N_SUPPORT_DIRECTIONS)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if (
                "dtheta" in cache.files
                and np.allclose(cache["dtheta"], condition, rtol=0.0, atol=1e-10)
                and cache["directions"].shape == directions.shape
                and np.allclose(cache["directions"], directions,
                                rtol=0.0, atol=1e-12)
            ):
                print(f"Using support-point cache: {cache_path}")
                return (
                    np.asarray(cache["support_points"]),
                    np.asarray(cache["support_success"]),
                )

    _, case = load_case("case33bw_ds")
    ec = case["errorcalculator"]
    update_parameters(ec, case["params"]["params_dict"], condition)
    points = np.full((N_SUPPORT_DIRECTIONS, 2), np.nan)
    success = np.zeros(N_SUPPORT_DIRECTIONS, dtype=bool)
    print(
        f"Computing {N_SUPPORT_DIRECTIONS} AC support points for "
        f"scenario {scenario:03d} ..."
    )
    for index, direction in enumerate(directions):
        point = ec.optimize_direction(-direction, in_approx=False)
        if point is not None and np.all(np.isfinite(point)):
            points[index] = point
            success[index] = True
        if (index + 1) % 30 == 0 or index + 1 == N_SUPPORT_DIRECTIONS:
            print(
                f"  support {index + 1}/{N_SUPPORT_DIRECTIONS}, "
                f"success={int(success[:index + 1].sum())}"
            )
    np.savez_compressed(
        cache_path,
        scenario=np.asarray(scenario),
        dtheta=np.asarray(condition),
        angles=angles,
        directions=directions,
        support_points=points,
        support_success=success,
        definition=np.asarray(
            "argmax direction^T x over the exact AC feasible model"
        ),
    )
    print(f"Saved support-point cache: {cache_path}")
    return points, success


def main():
    with np.load(POLYGON_BANK, allow_pickle=False) as bank:
        conditions = np.asarray(bank["dtheta"])
        A_initial = np.asarray(bank["A_initial"])
        b_initial = np.asarray(bank["b_initial"])
        A_updated = np.asarray(bank["A_updated"])
        b_updated = np.asarray(bank["b_updated"])

    with np.load(TRUTH_CACHE, allow_pickle=False) as truth:
        truth_conditions = np.asarray(truth["dtheta"])
        references = np.asarray(truth["x_ref"])

    with np.load(UPDATED_EVALUATION, allow_pickle=False) as evaluation:
        updated_coverage = np.asarray(evaluation["coverage"])

    if not np.allclose(conditions, truth_conditions, rtol=0.0, atol=1e-7):
        raise ValueError("The R1.1 polygon bank and common truth cache differ.")
    scenario = SELECTED_SCENARIO_ONE_BASED - 1
    if scenario < 0 or scenario >= len(conditions):
        raise IndexError(
            f"Selected condition {SELECTED_SCENARIO_ONE_BASED} is outside "
            f"the available 1..{len(conditions)} range."
        )
    reference = references[scenario]
    support_points, support_success = load_or_compute_support_points(
        scenario, conditions[scenario]
    )
    support_points = support_points[
        support_success & np.all(np.isfinite(support_points), axis=1)
    ]
    if len(support_points) < 3:
        raise RuntimeError("Fewer than three AC support optimizations succeeded.")
    unique_support = np.unique(np.round(support_points, 10), axis=0)
    sample_hull = unique_support[ConvexHull(unique_support).vertices]

    learned = {
        "Initial learned polygon": (
            halfspace_vertices(A_initial[scenario], b_initial[scenario]),
            "#2878B5",
        ),
        "Updated learned polygon": (
            halfspace_vertices(A_updated[scenario], b_updated[scenario]),
            "#D9534F",
        ),
    }

    initial_eval = R11_ROOT / "evaluation" / "lin_paper_initial_raw.npz"
    with np.load(initial_eval, allow_pickle=False) as evaluation:
        initial_coverage = np.asarray(evaluation["coverage"])
    coverage_by_title = {
        "Initial learned polygon": initial_coverage[scenario],
        "Updated learned polygon": updated_coverage[scenario],
    }

    all_points = [support_points, sample_hull]
    all_points.extend(item[0] for item in learned.values())
    extent = np.vstack(all_points)
    span = np.ptp(extent, axis=0)
    margin = 0.08 * max(float(span.max()), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)
    for panel, (axis, (title, (polygon, color))) in enumerate(
            zip(axes, learned.items())):
        hull_closed = closed(sample_hull)
        polygon_closed = closed(polygon)
        axis.fill(
            sample_hull[:, 0], sample_hull[:, 1], color="#BDBDBD",
            alpha=0.18, zorder=1,
        )
        axis.plot(
            hull_closed[:, 0], hull_closed[:, 1], color="black", lw=1.8,
            label="Hull of 360 AC support points", zorder=3,
        )
        axis.scatter(
            support_points[:, 0], support_points[:, 1], s=13,
            facecolors="white", edgecolors="#555555", linewidths=0.7,
            label="AC-feasible support points", zorder=4,
        )
        axis.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.16, zorder=2)
        axis.plot(
            polygon_closed[:, 0], polygon_closed[:, 1], color=color, lw=2.2,
            label=title, zorder=5,
        )
        axis.scatter(
            [reference[0]], [reference[1]], marker="*", s=90,
            color="#F0AD00", edgecolor="black", linewidth=0.6,
            label="Reference point", zorder=6,
        )
        axis.text(
            0.47, 0.77, coverage_text(coverage_by_title[title]),
            transform=axis.transAxes, va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "#B0B0B0", "alpha": 0.92},
        )
        axis.set_title(f"({chr(97 + panel)}) {title}")
        axis.set_xlabel("P (p.u.)")
        axis.set_ylabel("Q (p.u.)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(extent[:, 0].min() - margin, extent[:, 0].max() + margin)
        axis.set_ylim(extent[:, 1].min() - margin, extent[:, 1].max() + margin)
        axis.grid(ls="--", lw=0.6, alpha=0.28)
        axis.legend(loc="lower center", fontsize=8, framealpha=0.92)

    fig.suptitle(
        "R1.1 prediction versus 360-support-point AC hull\n"
        f"common test condition {SELECTED_SCENARIO_ONE_BASED} "
        f"(zero-based index {scenario})",
        fontsize=13,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "r11_prediction_vs_sample_hull_worst.png"
    svg_path = OUTPUT_DIR / "r11_prediction_vs_sample_hull_worst.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(
        f"condition={SELECTED_SCENARIO_ONE_BASED} "
        f"(zero-based index {scenario:03d})"
    )
    print(f"png={png_path}")
    print(f"svg={svg_path}")


if __name__ == "__main__":
    main()
