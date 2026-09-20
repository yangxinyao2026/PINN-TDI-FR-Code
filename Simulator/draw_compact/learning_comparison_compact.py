"""Export two standalone PDFs and a LaTeX cost table in pictures_compact/learning_comparison.

Display order: DDCL, DDNN (N=100, 500, 1000, 5000), PINN.
Source cache keys are unchanged. All table costs are in seconds.
Feasibility uses 50 conditions x 10 directions; radial coverage uses 50 x 100.
All observations are retained. Boxes show quartiles and orange median lines;
whiskers are the most extreme observations within 1.5 IQR. Jitter changes x only.
Coverage is predicted/true radial extent, and values above 100% are retained.

DDCL accounting follows the requested convention: offline data generation = 0,
offline training = 0, offline total = 0, online = the complete recorded cost
(215258.0977162 seconds). The historical cache calls that complete cost
"offline_total_seconds"; source data are unchanged and its lookup time is unused.
PINN/DDNN keep the recorded offline costs and first-call online timings. DDNN
label cost includes N training and 50 validation regions; PINN physics calls
are charged to training. No timing uncertainty was measured.

Times New Roman / STIX, pastel red/blue/green, black outlines, and horizontal grids
match the existing manuscript. Each PDF is 128.1 x 88 mm with editable vector text.
No titles, panel letters, explanatory footnotes, or combined figures are exported.
"""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import NullLocator
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_ROOT = PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
SUPERVISED_ROOT = COMPARISON_ROOT / "supervised" / "case33bw_ds" / "support_points"
LEARNING_ROOT = COMPARISON_ROOT / "learning_methods" / "case33bw_ds"
LIN_ROOT = LEARNING_ROOT / "lin_paper_2d_formal_validated_timed"
TIMING_PATH = (LEARNING_ROOT / "online_manuscript_protocol_current_hardware"
               / "online_manuscript_protocol_times.csv")
OUTPUT_ROOT = PROJECT_ROOT / "results" / "pictures_compact" / "learning_comparison"

METHODS = (
    {"key": "lin_paper_updated", "name": "DDCL",
     "color": "lightgreen", "root": LIN_ROOT},
    *({"key": f"supervised_n{n}", "name": f"DDNN ($N={n}$)", "n": n,
       "color": color, "root": SUPERVISED_ROOT, "timing_key": f"supervised_n{n}"}
      for n, color in [(100, "#D9ECF3"), (500, "#BFE0EC"),
                       (1000, "lightblue"), (5000, "#7FB6CE")]),
    {"key": "pinn", "name": "PINN", "color": "lightcoral",
     "root": SUPERVISED_ROOT, "timing_key": "pinn"},
)
METHOD_LABELS = [method["name"].replace(" (", "\n(") for method in METHODS]
ERROR_FIELDS = (
    ("feasibility_error", "feasibility_success", "feasibility"),
)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    rows = {row["method"]: row for row in records}
    if len(rows) != len(records):
        raise ValueError(f"Duplicate method names: {path}")
    return rows


def nonnegative_number(row, key):
    value = float(row[key])
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {key} for {row['method']}: {value}")
    return value


def load_results():
    summaries = {root: read_rows(root / "summary.csv")
                 for root in (SUPERVISED_ROOT, LIN_ROOT)}
    timing = read_rows(TIMING_PATH)
    with np.load(SUPERVISED_ROOT / "evaluation" / "test_truth.npz") as truth:
        conditions = truth["dtheta"].copy()
        evaluation_shape = truth["error_directions"].shape[:2]
        coverage_shape = truth["coverage_directions"].shape[:2]
        if evaluation_shape != (50, 10) or not truth["true_support_success"].all():
            raise ValueError("Expected 50 x 10 valid reference evaluations.")
        if (coverage_shape != (50, 100) or not truth["reference_success"].all()
                or not np.isfinite(truth["k_omega"]).all()
                or np.any(truth["k_omega"] <= 0)):
            raise ValueError("Expected 50 x 100 valid radial coverage references.")

    results = []
    for method in METHODS:
        row = summaries[method["root"]][method["key"]]
        record = dict(method)
        with np.load(method["root"] / "evaluation" / f"{method['key']}_raw.npz") as raw:
            np.testing.assert_array_equal(raw["dtheta"], conditions)
            for field, success_field, summary_prefix in ERROR_FIELDS:
                values = raw[field].copy()
                if values.shape != evaluation_shape or not raw[success_field].all():
                    raise ValueError(f"Incomplete evaluations: {method['key']} / {field}")
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError(f"Nonfinite or negative errors: {method['key']} / {field}")
                values = values.ravel()
                np.testing.assert_allclose(
                    [values.mean(), np.percentile(values, 95)],
                    [float(row[f"{summary_prefix}_mean"]), float(row[f"{summary_prefix}_p95"])],
                    rtol=1e-10, atol=0,
                )
                record[field] = values

            coverage = raw["coverage"].copy()
            if (coverage.shape != coverage_shape or not np.isfinite(coverage).all()
                    or np.any(coverage < 0)):
                raise ValueError(f"Invalid coverage evaluations: {method['key']}")
            if int(row["coverage_success_count"]) != coverage.size:
                raise ValueError(f"Coverage count mismatch: {method['key']}")
            np.testing.assert_allclose(
                [coverage.mean(), np.median(coverage)],
                [float(row["coverage_mean"]), float(row["coverage_median"])],
                rtol=1e-10, atol=0,
            )
            record["coverage_percent"] = coverage.ravel() * 100

        total = nonnegative_number(row, "offline_total_seconds")
        if method["key"] == "lin_paper_updated":
            training = nonnegative_number(row, "learning_seconds")
            preparation = total - training
            initial_labels = nonnegative_number(row, "initial_label_generation_seconds")
            if preparation < initial_labels:
                raise ValueError("DDCL source preparation cannot be below initial label cost.")
            record["initial_label_seconds"] = initial_labels
            record["adaptive_and_other_seconds"] = preparation - initial_labels
        else:
            training = nonnegative_number(row, "network_training_seconds")
            preparation = nonnegative_number(row, "label_generation_seconds")
        np.testing.assert_allclose(preparation + training, total, rtol=1e-12)
        if "n" in method:
            if (int(row["full_region_training_labels"]) != method["n"]
                    or int(row["full_region_validation_labels"]) != 50):
                raise ValueError(f"Expected N={method['n']} with 50 validation labels.")

        record["source_total_seconds"] = total
        if method["key"] == "lin_paper_updated":
            online_seconds = total
            preparation = 0.0
            training = 0.0
            total = 0.0
            record["online_basis"] = "all recorded computation assigned to online"
        else:
            time_row = timing[method["timing_key"]]
            if int(time_row["timed_online_calls"]) != 1:
                raise ValueError("Expected one recorded timed online call.")
            if time_row["model_loading_included"].lower() != "false":
                raise ValueError("The online protocol must exclude model loading.")
            if int(time_row["warmup_forward_calls"]) != 0:
                raise ValueError("Expected the first-call timing protocol.")
            online_seconds = nonnegative_number(time_row, "online_seconds")
            np.testing.assert_allclose(online_seconds * 1000,
                                       float(time_row["online_milliseconds"]), rtol=1e-12)
            record["online_basis"] = "recorded first timed call"
        record.update(preparation_seconds=preparation, training_seconds=training,
                      total_seconds=total, online_seconds=online_seconds)
        np.testing.assert_allclose(preparation + training, total, rtol=1e-12)
        results.append(record)
    return results


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
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
        "savefig.facecolor": "white",
    })


def save_pdf(figure, filename, subject):
    path = OUTPUT_ROOT / filename
    buffer = BytesIO()
    figure.savefig(buffer, format="pdf", metadata={
        "Title": filename.removesuffix(".pdf").replace("_", " "),
        "Subject": subject,
        "Creator": "Matplotlib",
    })
    plt.close(figure)
    try:
        path.write_bytes(buffer.getvalue())
    except PermissionError:
        # Windows viewers may lock an open PDF; keep the reviewed revision usable.
        path = path.with_stem(path.stem + "_revised")
        path.write_bytes(buffer.getvalue())
        print("Original PDF could not be overwritten; wrote the revised PDF instead.")
    print(f"PDF: {path}")


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(True, axis="y", which="major", color="0.7", alpha=0.3, linewidth=0.7)
    axis.set_xticks(np.arange(1, len(METHODS) + 1), METHOD_LABELS)
    axis.set_xlim(0.45, len(METHODS) + 0.55)
    axis.tick_params(axis="x", pad=5)


def draw_boxplot(axis, results, field, seed):
    """Match the manuscript's filled boxes and narrow jittered raw-point clouds."""
    data = [row[field] for row in results]
    positions = np.arange(1, len(results) + 1)
    boxes = axis.boxplot(
        data, positions=positions, widths=0.50, patch_artist=True,
        showfliers=False, whis=1.5, zorder=3,
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        medianprops={"color": "#FF7F0E", "linewidth": 1.3},
    )
    rng = np.random.default_rng(seed)
    for position, row, patch in zip(positions, results, boxes["boxes"]):
        patch.set_facecolor(to_rgba(row["color"], 0.6))
        values = row[field]
        # Every directional value is drawn. Jitter changes only horizontal position.
        jitter = rng.normal(0, 0.04, size=len(values))
        dense = len(values) > 1000
        axis.scatter(
            position + jitter, values, s=3 if dense else 5,
            color=row["color"], alpha=0.14 if dense else 0.30,
            edgecolors="black", linewidths=0.2, zorder=2,
            rasterized=False,
        )
    style_axis(axis)


def single_panel():
    width_scale = 0.70
    figure, axis = plt.subplots(figsize=(183 * width_scale / 25.4, 88 / 25.4))
    # Preserve physical margins and font sizes when reducing the page width.
    figure.subplots_adjust(left=0.10 / width_scale, right=1 - 0.02 / width_scale,
                           bottom=0.18, top=0.95)
    return figure, axis


def draw_feasibility_coverage(results):
    for field, ylabel, filename, seed in [
        ("feasibility_error", "Feasibility Error", "feasibility_error.pdf", 4201),
        ("coverage_percent", "Coverage Rate (%)", "coverage_rate.pdf", 4202),
    ]:
        figure, axis = single_panel()
        if field == "feasibility_error":
            if any(np.any(row[field] <= 0) for row in results):
                raise ValueError("A logarithmic feasibility axis requires positive errors.")
            axis.set_yscale("log")
            axis.set_ylim(1e-13, 1e-3)
            axis.set_yticks([1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3])
            axis.yaxis.set_minor_locator(NullLocator())
        else:
            axis.set_ylim(70, 120)
            axis.set_yticks([70, 80, 90, 100, 110, 120])
            axis.axhline(100, color="0.45", linestyle="--", linewidth=0.8, zorder=1)
        lower, upper = axis.get_ylim()
        if any(np.min(r[field]) < lower or np.max(r[field]) > upper for r in results):
            raise ValueError(f"Axis limits would crop {field} observations.")
        draw_boxplot(axis, results, field, seed)
        axis.set_ylabel(ylabel, labelpad=5)
        save_pdf(figure, filename,
                 f"{ylabel}: DDCL, DDNN N=100/500/1000/5000, PINN. All directional observations shown. "
                 "50 shared conditions, 10 feasibility directions or 100 coverage directions. "
                 "Boxes show quartiles and medians; whiskers use 1.5 IQR. "
                 "Feasibility is squared Euclidean projection distance; coverage is a radial ratio. "
                 "DDCL evaluates independently trained polygons at their trained conditions.")


def export_cost_table(results):
    """Export the same method order as the figures; all costs are in seconds."""
    def number(value, online=False):
        if value == 0:
            return "0"
        return f"{value:.5f}" if online and value < 1 else f"{value:.2f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Offline and online computational costs (s).}",
        r"\label{tab:learning_computation_costs}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Method &",
        r"\shortstack{Offline data\\generation} &",
        r"\shortstack{Offline\\training} &",
        r"\shortstack{Online\\computation} \\",
        r"\hline",
    ]
    for row in results:
        lines.append(
            f"{row['name']} & {number(row['preparation_seconds'])} & "
            f"{number(row['training_seconds'])} & {number(row['online_seconds'], online=True)}"
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path = OUTPUT_ROOT / "computation_costs.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"LaTeX: {path}")
    return path


def main():
    results = load_results()
    set_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    draw_feasibility_coverage(results)
    export_cost_table(results)
    for row in results:
        print(f"{row['name']}: offline data={row['preparation_seconds']:.6f} s; "
              f"training={row['training_seconds']:.6f} s; "
              f"offline total={row['total_seconds']:.6f} s; online={row['online_seconds']:.6f} s "
              f"({row['online_basis']})")



if __name__ == "__main__":
    main()
