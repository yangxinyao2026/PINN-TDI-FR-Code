"""Compare six NN configurations using the existing formal case33 evaluations.

Figure contract
---------------
Claim: all six configurations have very small squared feasibility errors, while
coverage varies more with activation than with adding a hidden layer in this run.
Design: two separate vertical boxplots with matching configuration order.
Each point is one operating-condition/direction observation, not a training seed.
Boxes show Q1--Q3 and the median. Whiskers reach the most extreme observations
within 1.5 IQR. All valid observations, including outliers, are drawn as points.
Normal jitter (SD 0.04) affects only the categorical axis and uses a fixed seed.
No significance tests are applied to these correlated within-model observations.
Feasibility is the stored squared Euclidean distance; coverage is k_P / k_Omega.
No clipping, rescaling of errors, subsampling, or pseudocounts are used.
Only coverage is converted from a ratio to percent.
Each configuration has 720 paired observations: 20 conditions x 36 directions.
Export: two 120 x 85 mm vector PDFs, using embedded Times New Roman throughout.

Run from any directory with Python, NumPy and Matplotlib:
    python plot_nn_architecture_distributions.py
Use --data-dir and --output-dir to override the defaults. Only PDF is exported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter
import numpy as np


SHARE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    SHARE_ROOT / "results" / "ds_proj_revise_V1" / "comparison" / "network"
    / "case33bw_ds"
)
DEFAULT_OUTPUT_DIR = (
    SHARE_ROOT / "results" / "pictures_compact"
    / "nn_architecture_distributions"
)
CONFIGS = (
    ("h128_act-relu", "[128]  ReLU"),
    ("h128_act-tanh", "[128]  Tanh"),
    ("h128_act-silu", "[128]  SiLU"),
    ("h128-128_act-relu", "[128, 128]  ReLU"),
    ("h128-128_act-tanh", "[128, 128]  Tanh"),
    ("h128-128_act-silu", "[128, 128]  SiLU"),
)
POSITIONS = np.array([1, 2, 3, 4.5, 5.5, 6.5])
COLORS = ("lightblue",) * 3 + ("lightgreen",) * 3


def load_distributions(data_dir: Path):
    """Check paired design and retain finite, successfully evaluated samples."""
    records = []
    reference_conditions = reference_directions = None
    for tag, label in CONFIGS:
        path = data_dir / tag / "formal_eval.npz"
        with np.load(path, allow_pickle=False) as raw:
            if str(raw["config"]) != tag:
                raise ValueError(f"Configuration mismatch: {path}")
            if str(raw["error_definition"]) != "squared_euclidean_distance":
                raise ValueError(f"Unexpected feasibility definition: {path}")
            if str(raw["coverage_definition"]) != "reference_point_radial_kP_over_kOmega":
                raise ValueError(f"Unexpected coverage definition: {path}")
            conditions, directions = raw["dthetas"], raw["eval_dirs"]
            if reference_conditions is None:
                reference_conditions, reference_directions = conditions.copy(), directions.copy()
            elif not (
                np.array_equal(conditions, reference_conditions)
                and np.array_equal(directions, reference_directions)
            ):
                raise ValueError("The six configurations do not share the same evaluation design.")
            feasibility, coverage = raw["feasibility"], raw["coverage"]
            expected_shape = (len(conditions), len(directions))
            if feasibility.shape != expected_shape or coverage.shape != expected_shape:
                raise ValueError(f"Unexpected metric shape: {path}")
            valid_feasibility = np.isfinite(feasibility) & raw["feasibility_success"]
            valid_coverage = (
                np.isfinite(coverage) & raw["radial_success"]
                & ~raw["membership_failure"][:, None]
                & raw["polygon_valid"][:, None]
            )
            f, c = feasibility[valid_feasibility], coverage[valid_coverage] * 100
            if not f.size or not c.size:
                raise ValueError(f"No valid observations: {path}")
            if np.any(f <= 0):
                raise ValueError("Log feasibility axis requires positive errors; no floor is applied.")
            records.append({"tag": tag, "label": label, "feasibility": f, "coverage": c})
            print(
                f"{tag}: feasibility n={f.size}/{feasibility.size}, "
                f"median={np.median(f):.6g}; coverage n={c.size}/{coverage.size}, "
                f"median={np.median(c):.3f}%"
            )
    return records, (len(reference_conditions), len(reference_directions))


def draw_distribution(ax, values, x, color, rng):
    ax.boxplot(
        [values], positions=[x], widths=0.5, whis=1.5,
        patch_artist=True, showfliers=False, manage_ticks=False,
        boxprops={"facecolor": to_rgba(color, 0.6), "edgecolor": "black", "linewidth": 0.75},
        medianprops={"color": "#D47732", "linewidth": 1.05, "zorder": 4},
        whiskerprops={"color": "black", "linewidth": 0.75},
        capprops={"color": "black", "linewidth": 0.75},
        zorder=3,
    )
    ax.scatter(
        rng.normal(x, 0.04, size=values.size), values,
        alpha=0.3, s=4, color=color, edgecolors="black", linewidths=0.2,
        zorder=2, rasterized=False,
    )


def configure_style():
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.linewidth": 0.65,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.65,
        "xtick.minor.width": 0.45,
        "pdf.fonttype": 42,
        "pdf.compression": 9,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.fallback": None,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def make_figure(records, metric):
    fig, ax = plt.subplots(figsize=(120 / 25.4, 85 / 25.4))
    fig.subplots_adjust(left=0.145, right=0.98, bottom=0.225, top=0.965)
    rng = np.random.default_rng(2306)
    for row, x, color in zip(records, POSITIONS, COLORS):
        draw_distribution(ax, row[metric], x, color, rng)
    ax.set_xlim(0.5, 7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="gray", alpha=0.3, linewidth=0.5)
    ax.axvline(3.75, color="gray", linestyle="--", alpha=0.3, linewidth=0.65)
    ax.tick_params(axis="both", which="major", length=3, pad=4)
    labels = [r["label"].rsplit("  ", 1) for r in records]
    ax.set_xticks(POSITIONS, [f"{activation}\n{hidden}" for hidden, activation in labels])
    ax.set_xlabel("NN architecture (activation function, hidden-layer neurons)",
                  fontsize=8.5, labelpad=7)
    values = np.concatenate([r[metric] for r in records])
    if metric == "feasibility":
        ax.set_yscale("log")
        ax.set_ylim(1e-13, 1e-9)
        ax.set_yticks(10.0 ** np.arange(-13, -8))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"1e{int(round(np.log10(value)))}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel("Feasibility Error", labelpad=6)
    else:
        coverage_min = min(70.0, np.floor(values.min() / 5) * 5)
        ax.set_ylim(coverage_min, 100)
        ax.set_yticks(np.arange(coverage_min, 101, 10))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_ylabel("Coverage Rate", labelpad=6)
    return fig


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    records, _ = load_distributions(args.data_dir)
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename in (
        ("feasibility", "feasibility_error_boxplot.pdf"),
        ("coverage", "coverage_boxplot.pdf"),
    ):
        fig = make_figure(records, metric)
        output = args.output_dir / filename
        metadata = {
            "Title": f"Neural-network architectures: {metric}",
            "Creator": "Matplotlib",
        }
        try:
            fig.savefig(output, format="pdf", metadata=metadata)
        except PermissionError:
            output = output.with_stem(output.stem + "_updated")
            fig.savefig(output, format="pdf", metadata=metadata)
            print("Original PDF is not writable; saved an updated copy instead.")
        plt.close(fig)
        print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
