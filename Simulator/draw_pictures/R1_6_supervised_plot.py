# -*- coding: utf-8 -*-
"""Plot the formal R1.6 comparison between supervised networks and the PINN."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Compatible ``python Simulator/draw_pictures/R1_6_supervised_plot.py`` Run directly.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from Simulator import PROJECT_ROOT


CASENAME = "case33bw_ds"
BASE_OUT = (
    PROJECT_ROOT
    / "results"
    / "ds_proj_revise_V1"
    / "comparison"
    / "supervised"
    / CASENAME
)
SUPPORT_OUT = BASE_OUT / "support_points"
# Direct execution plots the current support-point supervision results.
DEFAULT_LABEL_METHOD = "support"
ONLINE_TIMING = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "learning_methods" / CASENAME
    / "online_manuscript_protocol_current_hardware"
    / "online_manuscript_protocol_times.csv"
)

SUP_COLOR = "#2878B5"
PINN_COLOR = "#C82423"
ORACLE_COLOR = "#2E8B57"
GRID_ALPHA = 0.25

MAIN_TABLE_FIELDS = [
    "method", "full_region_training_labels", "feasibility_mean",
    "feasibility_p95", "optimality_mean", "optimality_p95",
    "average_radial_coverage", "radial_undercoverage",
    "radial_overcoverage", "reference_membership_rate",
    "offline_total_seconds", "physics_oracle_calls",
    "online_inference_seconds",
]

def result_dir(label_method):
    if label_method == "support":
        return SUPPORT_OUT
    raise ValueError(
        f"Currently, official experiments only retainsupportlabel,received: {label_method!r}")


def load_rows(out):
    path = out / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Data not found: {path}\nPlease run first main_revise1_6_supervised.py."
        )
    with path.open("r", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Summary file is empty: {path}")
    return rows


def apply_current_online_times(rows):
    """Replace old with unified original process retest valuessummarySteady state online time in."""
    if not ONLINE_TIMING.exists():
        raise FileNotFoundError(
            f"Unified online timing not found: {ONLINE_TIMING}\n"
            "Please run first main_revise1_1_online_cold_timing.py."
        )
    with ONLINE_TIMING.open("r", encoding="utf-8-sig") as file:
        timing = {
            row["method"]: row["online_seconds"]
            for row in csv.DictReader(file)
        }
    updated = []
    for row in rows:
        item = dict(row)
        method = item.get("method", "")
        if method in timing:
            item["online_inference_seconds"] = timing[method]
        updated.append(item)
    missing = [
        row.get("method", "") for row in updated
        if (row.get("method") == "pinn"
            or row.get("method", "").startswith("supervised_n"))
        and row.get("method") not in timing
    ]
    if missing:
        raise ValueError(f"Missing method for unified online timing: {missing}")
    return updated


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def with_publication_metric_names(row):
    """Add manuscript-facing names without changing legacy raw caches."""
    result = dict(row)
    result.update({
        "optimality_mean": number(row, "projection_optimality_mean"),
        "optimality_p95": number(row, "projection_optimality_p95"),
        # Original-manuscript definition: E[k_P/k_Omega].
        "average_radial_coverage": number(row, "coverage_mean"),
        "radial_undercoverage": number(row, "undercoverage_mean"),
        "radial_overcoverage": number(row, "overcoverage_mean"),
    })
    return result


def require_finite(values, name):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"indicator {name} Contains missing or non-finite values: {values}")
    return values


def find_method(rows, method, source):
    row = next((item for item in rows if item.get("method") == method), None)
    if row is None:
        raise ValueError(f"{source}No method found in {method!r}.")
    return row


def write_curated_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results table saved: {path}")
    return path


def export_main_results_table(rows, out):
    """Export text main table data: PINNwith all sizessupportsurveillance network."""
    pinn = find_method(rows, "pinn", out / "summary.csv")
    supervised = sorted(
        (row for row in rows if row.get("method", "").startswith("supervised_n")),
        key=lambda row: number(row, "full_region_training_labels"),
    )
    selected = [
        with_publication_metric_names(row)
        for row in [pinn, *supervised]
    ]
    return write_curated_csv(
        out / "R1_6_main_results_table.csv", selected, MAIN_TABLE_FIELDS)


def style_axis(ax, panel_title, ylabel, log_y=False):
    ax.set_title(panel_title, loc="left", fontsize=11)
    ax.set_xlabel("Complete training-region labels")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=GRID_ALPHA, which="both")


def plot_error_panel(ax, labels, supervised, pinn, oracle, mean_key, p95_key,
                     title, ylabel):
    sup_mean = require_finite(
        [number(row, mean_key) for row in supervised], mean_key
    )
    sup_p95 = require_finite(
        [number(row, p95_key) for row in supervised], p95_key
    )
    pinn_mean = number(pinn, mean_key)
    pinn_p95 = number(pinn, p95_key)

    ax.plot(
        labels,
        sup_mean,
        color=SUP_COLOR,
        marker="o",
        lw=1.8,
        label="Supervised: mean",
    )
    ax.plot(
        labels,
        sup_p95,
        color=SUP_COLOR,
        marker="^",
        lw=1.5,
        ls="--",
        label="Supervised: P95",
    )
    ax.axhline(
        pinn_mean,
        color=PINN_COLOR,
        lw=1.8,
        label="Physics-informed: mean",
    )
    ax.axhline(
        pinn_p95,
        color=PINN_COLOR,
        lw=1.5,
        ls="--",
        label="Physics-informed: P95",
    )
    if oracle is not None:
        oracle_mean = number(oracle, mean_key)
        oracle_p95 = number(oracle, p95_key)
        if np.isfinite(oracle_mean):
            ax.axhline(
                oracle_mean, color=ORACLE_COLOR, lw=1.6, ls="-.",
                label="Oracle label polygon: mean")
        if np.isfinite(oracle_p95):
            ax.axhline(
                oracle_p95, color=ORACLE_COLOR, lw=1.4, ls=":",
                label="Oracle label polygon: P95")
    style_axis(ax, title, ylabel, log_y=True)


def plot_single_metric(ax, labels, supervised, pinn, oracle, key, title, ylabel,
                       scale=1.0, log_y=False, percent=False,
                       supervised_label="Supervised FullNet",
                       pinn_label="Physics-informed FullNet"):
    values = scale * require_finite(
        [number(row, key) for row in supervised], key
    )
    pinn_value = scale * number(pinn, key)
    ax.plot(
        labels,
        values,
        color=SUP_COLOR,
        marker="o",
        lw=1.8,
        label=supervised_label,
    )
    ax.axhline(
        pinn_value,
        color=PINN_COLOR,
        ls="--",
        lw=1.6,
        label=pinn_label,
    )
    if oracle is not None:
        oracle_value = scale * number(oracle, key)
        if np.isfinite(oracle_value):
            ax.axhline(
                oracle_value,
                color=ORACLE_COLOR,
                ls="-.",
                lw=1.6,
                label="Oracle label polygon",
            )
    style_axis(ax, title, ylabel, log_y=log_y)
    if percent:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:g}%")


def main(label_method=DEFAULT_LABEL_METHOD):
    out = result_dir(label_method)
    rows = apply_current_online_times(load_rows(out))
    try:
        pinn = next(row for row in rows if row["method"] == "pinn")
    except StopIteration as exc:
        raise ValueError("summary.csv not found in pinn result.") from exc
    supervised = sorted(
        (row for row in rows if row["method"].startswith("supervised_n")),
        key=lambda row: number(row, "full_region_training_labels"),
    )
    if not supervised:
        raise ValueError("summary.csv not found in supervised_n* result.")
    oracle = next(
        (row for row in rows if row["method"].startswith("oracle_")), None
    )

    labels = require_finite(
        [number(row, "full_region_training_labels") for row in supervised],
        "full_region_training_labels",
    )

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.0))

    plot_error_panel(
        axes[0, 0],
        labels,
        supervised,
        pinn,
        oracle,
        "feasibility_mean",
        "feasibility_p95",
        "(a) Feasibility error",
        "Squared Euclidean distance",
    )
    plot_error_panel(
        axes[0, 1],
        labels,
        supervised,
        pinn,
        oracle,
        "projection_optimality_mean",
        "projection_optimality_p95",
        "(b) Optimality error",
        "Squared Euclidean distance",
    )
    plot_single_metric(
        axes[0, 2],
        labels,
        supervised,
        pinn,
        oracle,
        "coverage_mean",
        "(c) Average radial coverage",
        r"Mean $k_{P}/k_{\Omega}$ (%)",
        scale=100.0,
        percent=True,
    )
    axes[0, 2].axhline(100.0, color="0.35", ls=":", lw=1.2,
                       label="Exact coverage")

    plot_single_metric(
        axes[0, 3],
        labels,
        supervised,
        pinn,
        oracle,
        "undercoverage_mean",
        "(d) Mean undercoverage",
        r"$\mathbb{E}[\max(1-k_P/k_\Omega,0)]$ (%)",
        scale=100.0,
        percent=True,
    )

    plot_single_metric(
        axes[1, 0],
        labels,
        supervised,
        pinn,
        oracle,
        "overcoverage_mean",
        "(e) Mean overcoverage",
        r"$\mathbb{E}[\max(k_P/k_\Omega-1,0)]$ (%)",
        scale=100.0,
        percent=True,
    )
    axes[1, 0].set_ylim(bottom=-0.15)

    plot_single_metric(
        axes[1, 1],
        labels,
        supervised,
        pinn,
        None,
        "offline_total_seconds",
        "(f) Total offline computation",
        "Time (min)",
        scale=1.0 / 60.0,
        log_y=True,
    )
    plot_single_metric(
        axes[1, 2],
        labels,
        supervised,
        pinn,
        None,
        "physics_oracle_calls",
        "(g) Physics-model calls",
        "Number of optimization calls",
        log_y=True,
    )
    plot_single_metric(
        axes[1, 3],
        labels,
        supervised,
        pinn,
        None,
        "online_inference_seconds",
        "(h) Online neural inference (original protocol)",
        "Time per region (ms)",
        scale=1000.0,
    )

    # The horizontal axis reports the actual number of complete training labels.
    for ax in axes.flat:
        ax.set_xticks(labels)

    # Full legend for error plot; additional explanation for coverage plot100%Accurate baseline coverage.
    axes[0, 0].legend(fontsize=7.6, loc="best")
    handles, legend_labels = axes[0, 2].get_legend_handles_labels()
    axes[0, 2].legend(handles, legend_labels, fontsize=7.6, loc="best")
    axes[1, 0].legend(fontsize=7.6, loc="best")

    label_title = "Fixed-normal support-point labels"
    fig.suptitle(
        "Same-architecture supervised baseline versus physics-informed model\n"
        + label_title,
        fontsize=14,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.015,
        "Errors are squared Euclidean distances; all methods use the same "
        "50 test conditions and evaluation directions. Online time uses one "
        "end-to-end forward pass as in the original manuscript.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()

    out.mkdir(parents=True, exist_ok=True)
    png = out / "R1_6_supervised_comparison.png"
    pdf = out / "R1_6_supervised_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {png}")
    print(f"Vector saved: {pdf}")
    main_table = export_main_results_table(rows, out)
    return png, pdf, main_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-method",
        choices=("support",),
        default=DEFAULT_LABEL_METHOD,
        help=f"Current default read{DEFAULT_LABEL_METHOD}result.",
    )
    args = parser.parse_args()
    main(args.label_method)
