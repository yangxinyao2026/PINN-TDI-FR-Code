"""R1.1 representative learning-method comparison.

This script combines the formal independent-condition Lin-style experiment with the
same-architecture supervised/PINN experiment.  It deliberately reports the
actual physics-model calls and offline time because the three approaches use
different kinds of training information and a single ambiguous "sample
count" would not be a fair comparison.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ds_proj_revise_V1"
    / "comparison"
)
LEARNING_ROOT = RESULT_ROOT / "learning_methods" / "case33bw_ds"
LIN_SUMMARY = (
    LEARNING_ROOT / "lin_paper_2d_formal_validated_timed" / "summary.csv"
)
PINN_SUPERVISED_SUMMARY = (
    RESULT_ROOT / "supervised" / "case33bw_ds"
    / "support_points" / "summary.csv"
)
ONLINE_TIMING = (
    LEARNING_ROOT / "online_manuscript_protocol_current_hardware"
    / "online_manuscript_protocol_times.csv"
)
OUTPUT_CSV = LEARNING_ROOT / "R1_1_learning_comparison_table.csv"
OUTPUT_PNG = LEARNING_ROOT / "R1_1_learning_comparison.png"
OUTPUT_PDF = LEARNING_ROOT / "R1_1_learning_comparison.pdf"


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing formal result: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["method"]: row for row in rows}


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _online_times() -> dict[str, float]:
    """Read the current hardware retest results of the unified original process and prohibit the return to the old steady-state time.."""
    rows = _read_rows(ONLINE_TIMING)
    return {
        method: _number(row, "online_seconds", np.nan)
        for method, row in rows.items()
    }


def _build_records() -> list[dict[str, object]]:
    common = _read_rows(PINN_SUPERVISED_SUMMARY)
    lin = _read_rows(LIN_SUMMARY)
    online = _online_times()

    specifications = [
        {
            "name": "PINN",
            "row": common["pinn"],
            "information": "Physics-informed directional optimization",
            "model": "Nonlinear NN: theta -> (A, b)",
            "call_key": "physics_oracle_calls",
            "complete_train": 0,
            "complete_validation": 0,
            "points_generated": 0,
            "points_retained": 0,
            # PINN samples operating conditions on the fly and has no fixed
            # finite training-condition data set.  Report exposures instead.
            "conditions_used": "",
            "condition_exposures": int(
                _number(common["pinn"], "condition_exposures")
            ),
            "online_method": "pinn",
        },
        {
            "name": "Supervised (N=50)",
            "row": common["supervised_n50"],
            "information": "Complete 36-direction AC support-point labels",
            "model": "Boundary-point NN: theta -> 36 (P, Q), then convex hull",
            "call_key": "physics_oracle_calls",
            "complete_train": int(
                _number(
                    common["supervised_n50"],
                    "full_region_training_labels",
                )
            ),
            "complete_validation": int(
                _number(
                    common["supervised_n50"],
                    "full_region_validation_labels",
                )
            ),
            "points_generated": 0,
            "points_retained": 0,
            "conditions_used": int(
                _number(common["supervised_n50"], "unique_training_conditions")
            ),
            "condition_exposures": int(
                _number(common["supervised_n50"], "condition_exposures")
            ),
            "online_method": "supervised_n50",
        },
        {
            "name": "Lin-style independent polygons (updated)",
            "row": lin["lin_paper_updated"],
            "information": "Pointwise labels plus violation-driven updating",
            "model": "50 independently learned static (A, b) polytopes",
            "call_key": "high_level_physics_calls",
            "complete_train": 0,
            "complete_validation": 0,
            "points_generated": int(
                _number(
                    lin["lin_paper_updated"],
                    "point_labels_generated",
                )
            ),
            "points_retained": int(
                _number(
                    lin["lin_paper_updated"],
                    "point_labels_final_training",
                )
            ),
            "conditions_used": int(
                _number(lin["lin_paper_updated"], "training_conditions")
            ),
            "condition_exposures": "",
            "online_method": "r11_ac_validated_updated",
        },
    ]

    records: list[dict[str, object]] = []
    for specification in specifications:
        name = specification["name"]
        row = specification["row"]
        under = _number(row, "undercoverage_mean")
        over = _number(row, "overcoverage_mean")
        records.append(
            {
                "method": name,
                "training_information": specification["information"],
                "model_form": specification["model"],
                "complete_region_training_labels":
                    specification["complete_train"],
                "complete_region_validation_labels":
                    specification["complete_validation"],
                "point_labels_generated":
                    specification["points_generated"],
                "point_labels_retained":
                    specification["points_retained"],
                "operating_conditions_used":
                    specification["conditions_used"],
                "condition_exposures":
                    specification["condition_exposures"],
                "true_domain_optimization_calls": int(
                    _number(row, specification["call_key"])
                ),
                "offline_total_seconds": _number(row, "offline_total_seconds"),
                "online_inference_seconds": online[specification["online_method"]],
                "feasibility_mean": _number(row, "feasibility_mean"),
                "feasibility_p95": _number(row, "feasibility_p95"),
                # Publication name is "optimality error"; the raw source key
                # is retained only for backward-compatible result caches.
                "optimality_mean": _number(row, "projection_optimality_mean"),
                "optimality_p95": _number(row, "projection_optimality_p95"),
                # Same definition as the original manuscript and
                # approximate_polygon_coverage.py: E[k_P/k_Omega].
                "average_radial_coverage": _number(row, "coverage_mean"),
                "radial_undercoverage": under,
                "radial_overcoverage": over,
                "symmetric_radial_error": under + over,
                "test_conditions": int(_number(row, "test_conditions", 50.0)),
                "evaluation_directions": int(_number(row, "coverage_success_count", 5000.0)),
            }
        )
    return records


def _write_table(records: list[dict[str, object]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, np.finfo(float).tiny)


def _draw(records: list[dict[str, object]]) -> None:
    names = [str(row["method"]) for row in records]
    short_names = ["PINN", "Supervised\n(N=50)", "Lin-style\nupdated"]
    x = np.arange(len(records))
    colors = ["#d62728", "#1f77b4", "#ff9f1c", "#2ca02c"]
    colors = colors[:len(records)]

    figure, axes = plt.subplots(2, 3, figsize=(15.8, 8.3), constrained_layout=False)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.16, wspace=0.27, hspace=0.37)

    metric_pairs = [
        ("feasibility_mean", "feasibility_p95", "(a) Feasibility error"),
        ("optimality_mean", "optimality_p95", "(b) Optimality error"),
    ]
    for axis, (mean_key, p95_key, title) in zip(axes[0, :2], metric_pairs):
        mean = _positive(np.array([float(row[mean_key]) for row in records]))
        p95 = _positive(np.array([float(row[p95_key]) for row in records]))
        width = 0.36
        axis.bar(x - width / 2, mean, width, color=colors, edgecolor="black", linewidth=0.5, label="Mean")
        axis.bar(x + width / 2, p95, width, color=colors, alpha=0.48, hatch="//", edgecolor="black", linewidth=0.5, label="P95")
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel("Squared Euclidean distance")
        axis.set_xticks(x, short_names)
        axis.grid(axis="y", which="both", alpha=0.24)
        axis.legend(fontsize=8)

    axis = axes[0, 2]
    coverage = 100.0 * np.array([float(row["average_radial_coverage"]) for row in records])
    under = 100.0 * np.array([float(row["radial_undercoverage"]) for row in records])
    over = 100.0 * np.array([float(row["radial_overcoverage"]) for row in records])
    width = 0.25
    axis.bar(x - width, coverage, width, color=colors, edgecolor="black", linewidth=0.5, label="Average coverage")
    axis.bar(x, under, width, color=colors, alpha=0.66, hatch="//", edgecolor="black", linewidth=0.5, label="Undercoverage")
    axis.bar(x + width, over, width, color=colors, alpha=0.38, hatch="..", edgecolor="black", linewidth=0.5, label="Overcoverage")
    axis.axhline(100.0, color="0.35", ls=":", lw=1.1)
    axis.set_title("(c) Radial coverage metrics")
    axis.set_ylabel("Directional radial metric (%)")
    axis.set_xticks(x, short_names)
    axis.grid(axis="y", alpha=0.24)
    axis.legend(fontsize=8)

    lower_metrics = [
        ("offline_total_seconds", 1.0 / 60.0, "(d) Total offline computation", "Time (min)", True),
        ("true_domain_optimization_calls", 1.0, "(e) True-domain optimization calls", "Number of calls", True),
        ("online_inference_seconds", 1000.0, "(f) Online region inference", "Time per region (ms)", False),
    ]
    for axis, (key, scale, title, ylabel, log_scale) in zip(axes[1], lower_metrics):
        values = np.array([float(row[key]) * scale for row in records])
        axis.bar(x, values, width=0.62, color=colors, edgecolor="black", linewidth=0.5)
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, short_names)
        axis.grid(axis="y", which="both", alpha=0.24)

    figure.suptitle(
        "R1.1 Representative learning-method comparison on case33bw_ds\n"
        "Lin-style learns one static polygon per sampled operating condition",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.045,
        "Training information is method-specific; true-domain optimization calls and total offline time are reported. "
        "The Lin-style independent-condition method uses pointwise labels, the supervised model uses complete 36-direction support-point labels, "
        "and PINN uses directional physics optimization during training. Online time follows the original manuscript's "
        "single end-to-end forward-pass protocol (model loading excluded).",
        ha="center",
        va="center",
        fontsize=9,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    records = _build_records()
    _write_table(records)
    _draw(records)
    print("R1.1 formal comparison completed")
    print(f"Table: {OUTPUT_CSV}")
    print(f"Figure: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
