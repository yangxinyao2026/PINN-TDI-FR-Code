# -*- coding: utf-8 -*-
"""Plot the case118_ts + multi-case33bw_ds downstream experiment."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from Simulator import PROJECT_ROOT


ROOT = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "downstream" / "case118_ts_multi_case33bw_ds"
)
DEFAULT_N_DSO = 10


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing formal results: {path}\n"
            "Run main_revise1_5and3_2downstream_large.py in formal mode first."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def heatmap(ax, matrix, title, fmt=".3f", cmap="viridis"):
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("TSO load factor")
    ax.set_ylabel("Heterogeneous DSO profile")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "–" if not np.isfinite(value) else format(value, fmt)
            ax.text(col, row, text, ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-dso", type=int, default=DEFAULT_N_DSO)
    args = parser.parse_args()

    out = ROOT / f"formal_{args.n_dso}dso_5profiles_3loads"
    rows = read_rows(out / "formal_summary.csv")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    profiles = sorted({row["profile"] for row in rows})
    loads = sorted({number(row, "ts_load_factor") for row in rows})
    lookup = {(row["profile"], number(row, "ts_load_factor")): row for row in rows}

    metrics = (
        ("value_retention_percent", "Value retention (%)", ".2f", "YlGnBu"),
        ("pinn_true_relative_cost_gap_percent", "PINN-to-True cost gap (%)", ".4f", "YlOrRd"),
        ("pinn_true_pg_l1_mw", "Generator dispatch deviation L1 (MW)", ".3f", "PuBu"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (key, title, fmt, cmap) in zip(axes, metrics):
        matrix = np.array([
            [number(lookup[(profile, load)], key) for load in loads]
            for profile in profiles
        ])
        heatmap(ax, matrix, title, fmt=fmt, cmap=cmap)
        ax.set_xticks(range(len(loads)), [f"{value:.2f}" for value in loads])
        ax.set_yticks(range(len(profiles)), profiles)
    fig.suptitle(f"case118_ts with {args.n_dso} case33bw_ds systems", fontsize=15)
    fig.savefig(figures / "multi_dso_downstream_heatmaps.png", dpi=300)
    fig.savefig(figures / "multi_dso_downstream_heatmaps.pdf")
    fig.savefig(figures / "multi_dso_downstream_heatmaps.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for region, color, label in (
        ("base", "#6b7280", "No flexibility"),
        ("pinn", "#2563eb", "PINN polygons"),
        ("true", "#dc2626", "True DSO models"),
    ):
        means = []
        spreads = []
        for load in loads:
            values = np.array([
                number(row, f"{region}_cost") for row in rows
                if np.isclose(number(row, "ts_load_factor"), load)
            ])
            means.append(np.nanmean(values))
            spreads.append(np.nanstd(values))
        axes[0].errorbar(loads, means, yerr=spreads, marker="o", color=color, label=label)
    axes[0].set_xlabel("TSO load factor")
    axes[0].set_ylabel("Dispatch cost")
    axes[0].set_title("Mean cost across five DSO profiles")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    gaps = np.array([number(row, "pinn_true_relative_cost_gap_percent") for row in rows])
    retention = np.array([number(row, "value_retention_percent") for row in rows])
    pg_l1 = np.array([number(row, "pinn_true_pg_l1_mw") for row in rows])
    scatter = axes[1].scatter(gaps, retention, c=pg_l1, cmap="viridis", s=65)
    axes[1].set_xlabel("PINN-to-True cost gap (%)")
    axes[1].set_ylabel("Value retention (%)")
    axes[1].set_title("Economic impact and dispatch deviation")
    axes[1].grid(alpha=0.25)
    colorbar = plt.colorbar(scatter, ax=axes[1])
    colorbar.set_label("Generator dispatch deviation L1 (MW)")
    fig.savefig(figures / "multi_dso_cost_and_value.png", dpi=300)
    fig.savefig(figures / "multi_dso_cost_and_value.pdf")
    fig.savefig(figures / "multi_dso_cost_and_value.svg")
    plt.close(fig)
    print(f"Figures saved to: {figures}")


if __name__ == "__main__":
    main()
