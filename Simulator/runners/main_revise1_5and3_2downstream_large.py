# -*- coding: utf-8 -*-
"""R1.5/R3.2: large-TSO, multi-DSO downstream validation.

This script compares four 27-DSO case118 transmission-system versions:

* v1: 27 independent copies of case33bw_ds;
* v2: 27 independent copies of case533mt_hi_ds.
* v3: 27 independent copies of case118zh_ds, scaled at the TSO/DSO interface;
* v4: 27 independent copies of case36real_3phase_ds.

The count follows the reviewer protocol: at the original, unscaled case118_ts
load, the greedy physical screen accepts m=29 case33 systems and n=27 case533
systems, hence k=min(m,n)=27. No transmission-load scaling alpha is used.

Each version compares three TSO dispatch formulations:

* Base: every DSO is fixed at its no-flexibility reference point;
* PINN: every DSO is represented by the polygon predicted by the original PINN;
* True: every DSO is represented by its complete nonlinear branch-flow model.

The host-load decomposition is performed once at the nominal DSO condition.
Changing a DSO operating condition therefore never cancels or redefines the
native TSO load.  PCC voltages are fixed at 1.0 p.u., consistently with the
PINN input definition used in the paper.

Execution
---------
Use ``--version v1|v2|v3|v4|all`` to select the immutable DSO case/count/PCC set
and ``--mode timing|pilot|formal|multistart|all`` to select the work stage.
Direct execution runs the two newly added versions (v3 and v4): five DSO
coverage profiles at the original TSO load and no additional multi-start audit.
Every version writes to its own directory. ``--mode all`` runs pilot, formal,
and multistart for one selected version.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import torch

# Allow both ``python -m ...`` and direct execution from PyCharm/PowerShell.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from Simulator import PROJECT_ROOT
from Simulator.cases import TD_case, DS_case_3phase
from Simulator.runners import main_revise1_5and3_2downstream as legacy
from Simulator.solver_environment import prepare_ipopt


TS_CASE = "case118_ts"
PCC_VOLTAGE = 1.0
SOLVER_NAME = "ipopt"

# PCC candidates are nonzero-load PQ buses ranked by nominal apparent load.
# Each added PCC is accepted only if the Base case remains optimal with V_PCC fixed at 1.0 p.u.
PCC_SELECTION_FILE = "pcc_feasibility_screen.csv"

# The unmodified IEEE 118-bus generator limits make the no-flexibility Base
# infeasible at 1.10 loading, whereas PINN and True remain feasible.  The
# formal three-cost comparison therefore uses a 5% high-load condition; all
# five representative Base profiles pass the same economic-optimality check.
FORMAL_TS_LOAD_FACTORS = (1.00,)
FORMAL_PERCENTILES = (10, 30, 50, 70, 90)
# Direct PyCharm execution performs the requested solver-time comparison.
# Only solver.solve() is timed; construction and PINN inference are excluded.
DEFAULT_MODE = "timing"
DEFAULT_VERSION = "new"
INITIAL_VERSION = "v1"
DISAGGREGATION_TOL = 1e-8
TS_MODEL_VERSION = "standard_ac_ybus_original_q_slack_v_2026_09"
MULTISTART_COUNT = 5
# Low-, central-, and high-coverage profiles at the original TSO load.
REPRESENTATIVE_MULTISTART = ((0, 1.00), (2, 1.00), (4, 1.00))

CASE33_CONDITION_DIR = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "downstream" / "case4gs_ts_case33bw_ds"
    / "operating_grid_random_dtheta"
)
CASE533_MODEL_DIR = (
    PROJECT_ROOT / "results" / "ds_proj_paper" / "case533mt_hi_ds"
    / "A(36,2)_type3(9-36)_lr1(1e-4)_lr2(1e-4)_rate(1e-4)"
)
CASE533_COVERAGE_PATH = (
    CASE533_MODEL_DIR / "figures" / "comparison" / "feasible" / "contrast"
    / "comparison_AnalyticalPolygon" / "coverage_data_m36.npz"
)
CASE118_MODEL_DIR = (
    PROJECT_ROOT / "results" / "ds_proj_paper" / "case118zh_ds"
    / "A(36,2)_type3(97, 107, 109, 80, 63, 31)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)"
)
CASE118_COVERAGE_PATH = (
    CASE118_MODEL_DIR / "figures" / "comparison" / "feasible" / "contrast"
    / "comparison_AnalyticalPolygon" / "coverage_data_m36.npz"
)
CASE36_MODEL_DIR = (
    PROJECT_ROOT / "results" / "ds_proj_paper" / "case36real_3phase_ds"
    / "A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)"
)
CASE36_COVERAGE_PATH = (
    CASE36_MODEL_DIR / "figures" / "comparison" / "feasible" / "contrast"
    / "comparison_AnalyticalPolygon" / "coverage_data_m36.npz"
)
VERSION_OUT_ROOT = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "downstream" / "case118_ts_equal_count_no_alpha"
)

# MATPOWER bus numbers obtained by the no-alpha greedy screen. Each candidate
# must leave nonnegative native host P/Q and keep all five DSO percentile Base
# cases optimal at the original case118_ts load. The full safe counts are
# m=29 and n=27; both comparison versions retain k=27 PCCs.
CASE33_MAX_PCC_BUS_NUMBERS = (
    60, 78, 11, 82, 45, 95, 79, 88, 75, 106,
    3, 41, 35, 39, 67, 16, 53, 33, 29, 86,
    115, 2, 51, 28, 20, 44, 21, 109, 114,
)
CASE533_MAX_PCC_BUS_NUMBERS = (
    60, 78, 11, 82, 45, 95, 79, 88, 75, 106,
    3, 41, 35, 39, 67, 16, 53, 33, 29, 86,
    115, 2, 7, 51, 28, 20, 44,
)
# V3/V4 are not required to share PCCs.  Each configured 27-bus set is checked
# with the complete Base AC OPF (including V_PCC=1.0 p.u.) for all five DSO
# profiles.  V3 uses a uniform interface divisor of 4.0 because smaller tested
# values up to 3.0 caused one Base case to hit Ipopt's iteration limit; V4 is
# small enough to require no interface scaling.
CASE118_PCC_BUS_NUMBERS = (
    60, 78, 11, 82, 45, 95, 79, 88, 75, 106, 3, 41, 35, 39,
    67, 16, 53, 33, 29, 86, 115, 2, 51, 28, 20, 44, 21,
)
CASE36_PCC_BUS_NUMBERS = (
    60, 78, 11, 82, 45, 95, 79, 88, 75, 106, 3, 41, 35, 39,
    67, 16, 53, 33, 29, 86, 115, 2, 7, 51, 28, 20, 44,
)
EQUAL_DSO_COUNT = min(len(CASE33_MAX_PCC_BUS_NUMBERS),
                      len(CASE533_MAX_PCC_BUS_NUMBERS))

VERSION_CONFIGS = {
    "v1": {
        "label": "version1_27case33",
        "ds_case": "case33bw_ds",
        "n_dso": EQUAL_DSO_COUNT,
        "pcc_bus_numbers": CASE33_MAX_PCC_BUS_NUMBERS[:EQUAL_DSO_COUNT],
        "model_dir": legacy.PAPER_MODEL_DIR,
        "condition_source": CASE33_CONDITION_DIR,
        "interface_divisor": 1.0,
        "is_3phase": False,
    },
    "v2": {
        "label": "version2_27case533",
        "ds_case": "case533mt_hi_ds",
        "n_dso": EQUAL_DSO_COUNT,
        "pcc_bus_numbers": CASE533_MAX_PCC_BUS_NUMBERS,
        "model_dir": CASE533_MODEL_DIR,
        "condition_source": CASE533_COVERAGE_PATH,
        "interface_divisor": 1.0,
        "is_3phase": False,
    },
    "v3": {
        "label": "version3_27case118_interface_div4",
        "ds_case": "case118zh_ds",
        "n_dso": 27,
        "pcc_bus_numbers": CASE118_PCC_BUS_NUMBERS,
        "model_dir": CASE118_MODEL_DIR,
        "condition_source": CASE118_COVERAGE_PATH,
        "interface_divisor": 4.0,
        "is_3phase": False,
    },
    "v4": {
        "label": "version4_27case36_3phase",
        "ds_case": "case36real_3phase_ds",
        "n_dso": 27,
        "pcc_bus_numbers": CASE36_PCC_BUS_NUMBERS,
        "model_dir": CASE36_MODEL_DIR,
        "condition_source": CASE36_COVERAGE_PATH,
        "interface_divisor": 1.0,
        "is_3phase": True,
    },
}

# Runtime values are set by configure_version() before any model is built.
EXPERIMENT_VERSION = INITIAL_VERSION
DS_CASE = VERSION_CONFIGS[INITIAL_VERSION]["ds_case"]
DEFAULT_N_DSO = int(VERSION_CONFIGS[INITIAL_VERSION]["n_dso"])
PILOT_N_DSO = (5, DEFAULT_N_DSO)
SOURCE_RESULT_DIR = VERSION_CONFIGS[INITIAL_VERSION]["condition_source"]
OUT_ROOT = VERSION_OUT_ROOT / VERSION_CONFIGS[INITIAL_VERSION]["label"]


def interface_divisor() -> float:
    return float(VERSION_CONFIGS[EXPERIMENT_VERSION]["interface_divisor"])


def interface_scale() -> float:
    return 1.0 / interface_divisor()


def is_three_phase() -> bool:
    return bool(VERSION_CONFIGS[EXPERIMENT_VERSION].get("is_3phase", False))


def load_ds_case() -> dict:
    if is_three_phase():
        return DS_case_3phase.case36real_3phase_ds()
    return getattr(TD_case, DS_CASE)(root_voltage=PCC_VOLTAGE)


def configure_version(version: str) -> dict:
    """Activate one complete, non-mixable downstream experiment version."""
    global EXPERIMENT_VERSION, DS_CASE, DEFAULT_N_DSO, PILOT_N_DSO
    global SOURCE_RESULT_DIR, OUT_ROOT
    config = VERSION_CONFIGS[version]
    EXPERIMENT_VERSION = version
    DS_CASE = str(config["ds_case"])
    DEFAULT_N_DSO = int(config["n_dso"])
    PILOT_N_DSO = (5, DEFAULT_N_DSO)
    SOURCE_RESULT_DIR = config["condition_source"]
    OUT_ROOT = VERSION_OUT_ROOT / str(config["label"])
    model_dir = Path(config["model_dir"])
    legacy.PAPER_MODEL_DIR = model_dir
    legacy.PRETRAIN_WEIGHTS = model_dir / "pretrainnet_weights.pth"
    legacy.FULLNET_WEIGHTS = model_dir / "fullnet_weights_feasible.pth"
    return config


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_solver_time_statistics(path: Path, rows: list[dict]) -> None:
    """Write solver-only timing statistics and paired True/PINN speedup."""
    statistics = []
    for region in ("base", "pinn", "true"):
        values = np.asarray([
            float(row[f"{region}_solve_time_s"])
            for row in rows
            if bool(row[f"{region}_optimal"])
            and np.isfinite(float(row[f"{region}_solve_time_s"]))
        ], dtype=float)
        statistics.append({
            "experiment_version": EXPERIMENT_VERSION,
            "ds_case": DS_CASE,
            "n_dso": DEFAULT_N_DSO,
            "region": region,
            "attempted_scenarios": len(rows),
            "optimal_scenarios": int(values.size),
            "mean_solve_time_s": float(np.mean(values)) if values.size else np.nan,
            "median_solve_time_s": (
                float(np.median(values)) if values.size else np.nan
            ),
            "std_solve_time_s": float(np.std(values)) if values.size else np.nan,
            "min_solve_time_s": float(np.min(values)) if values.size else np.nan,
            "p95_solve_time_s": (
                float(np.percentile(values, 95)) if values.size else np.nan
            ),
            "max_solve_time_s": float(np.max(values)) if values.size else np.nan,
            "timing_scope": "solver.solve wall-clock only",
        })
    _write_csv(path, statistics)

    paired = [
        row for row in rows
        if bool(row["pinn_optimal"]) and bool(row["true_optimal"])
        and np.isfinite(float(row["pinn_solve_time_s"]))
        and np.isfinite(float(row["true_solve_time_s"]))
        and float(row["pinn_solve_time_s"]) > 0.0
    ]
    ratios = np.asarray([
        float(row["true_solve_time_s"]) / float(row["pinn_solve_time_s"])
        for row in paired
    ], dtype=float)
    mean_pinn = (
        float(np.mean([float(row["pinn_solve_time_s"]) for row in paired]))
        if paired else np.nan
    )
    mean_true = (
        float(np.mean([float(row["true_solve_time_s"]) for row in paired]))
        if paired else np.nan
    )
    _write_csv(path.with_name("solver_time_speedup.csv"), [{
        "experiment_version": EXPERIMENT_VERSION,
        "ds_case": DS_CASE,
        "n_dso": DEFAULT_N_DSO,
        "paired_optimal_scenarios": len(paired),
        "mean_pinn_solve_time_s": mean_pinn,
        "mean_true_solve_time_s": mean_true,
        "ratio_of_mean_times_true_over_pinn": (
            mean_true / mean_pinn
            if np.isfinite(mean_true) and np.isfinite(mean_pinn)
            and mean_pinn > 0.0 else np.nan
        ),
        "median_paired_speedup_true_over_pinn": (
            float(np.median(ratios)) if ratios.size else np.nan
        ),
        "timing_scope": "solver.solve wall-clock only",
    }])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_relative_path(path: Path) -> str:
    """Return a public repository path without exposing a local source path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _formal_directory_name(n_dso: int, timing_only: bool = False) -> str:
    prefix = "solver_timing" if timing_only else "formal"
    load_word = "load" if len(FORMAL_TS_LOAD_FACTORS) == 1 else "loads"
    return (
        f"{prefix}_{n_dso}dso_{len(FORMAL_PERCENTILES)}profiles_"
        f"{len(FORMAL_TS_LOAD_FACTORS)}{load_word}"
    )


def save_experiment_config(
    ipopt: dict[str, str], n_dso: int, timing_only: bool = False
) -> Path:
    """Persist enough provenance to identify the formal server/model run."""
    tsppc = getattr(TD_case, TS_CASE)()
    nodes = load_screened_pcc_nodes(tsppc, n_dso)
    branch = np.asarray(tsppc["branch"], dtype=float)
    bus = np.asarray(tsppc["bus"], dtype=float)
    config = {
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_version_label": VERSION_CONFIGS[EXPERIMENT_VERSION]["label"],
        "python_executable": Path(sys.executable).name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "ipopt_executable": Path(ipopt["executable"]).name,
        "ipopt_version": ipopt["version"],
        "ts_case": TS_CASE,
        "ds_case": DS_CASE,
        "ts_model_version": TS_MODEL_VERSION,
        "ts_bus_count": int(len(tsppc["bus"])),
        "ts_branch_count": int(len(branch)),
        "ts_generator_count": int(len(tsppc["gen"])),
        "positive_rate_a_branches": int(np.count_nonzero(branch[:, 5] > 0.0)),
        "pcc_voltage_pu": PCC_VOLTAGE,
        "pcc_indices_zero_based": [int(node) for node in nodes],
        "pcc_bus_numbers": [int(bus[node, 0]) for node in nodes],
        "pcc_selection_policy": (
            "case-specific PQ-bus sets; V3/V4 need not share identical PCCs. "
            "Each configured 27-bus set is jointly verified by the Base AC "
            "OPF for all five DSO profiles with V_PCC fixed at 1.0 p.u."
        ),
        "n_dso": int(n_dso),
        "interface_divisor": interface_divisor(),
        "interface_scale": interface_scale(),
        "interface_scaling_definition": (
            "Only TSO-visible P/Q exchange is divided by interface_divisor; "
            "the DSO internal per-unit model and PINN inputs are unchanged."
        ),
        "true_model_initialization_index": 1 if is_three_phase() else 0,
        "formal_tso_load_factors": list(FORMAL_TS_LOAD_FACTORS),
        "formal_coverage_percentiles": list(FORMAL_PERCENTILES),
        "multistart_count": MULTISTART_COUNT,
        "representative_multistart": [
            {
                "profile": f"target_P{FORMAL_PERCENTILES[index]}",
                "ts_load_factor": factor,
            }
            for index, factor in REPRESENTATIVE_MULTISTART
        ],
        "pretrain_weights": _repository_relative_path(legacy.PRETRAIN_WEIGHTS),
        "pretrain_weights_sha256": _sha256(legacy.PRETRAIN_WEIGHTS),
        "fullnet_weights": _repository_relative_path(legacy.FULLNET_WEIGHTS),
        "fullnet_weights_sha256": _sha256(legacy.FULLNET_WEIGHTS),
        "coverage_condition_source": _repository_relative_path(SOURCE_RESULT_DIR),
        "timing_scope": (
            "solver.solve only; excludes model construction, PINN inference, "
            "data loading, and DSO disaggregation"
        ),
    }
    out = OUT_ROOT / _formal_directory_name(n_dso, timing_only)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "experiment_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return path


def _safe_percent(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if abs(denominator) <= 1e-12:
        return np.nan
    return 100.0 * numerator / denominator


def validate_pcc_nodes(
    tsppc: dict, n_dso: int, nodes: tuple[int, ...]
) -> tuple[int, ...]:
    if n_dso < 1 or n_dso > len(nodes):
        raise ValueError(f"n_dso must lie in [1, {len(nodes)}]")
    nodes = tuple(nodes[:n_dso])
    bus = np.asarray(tsppc["bus"])
    for node in nodes:
        if node >= len(bus):
            raise ValueError(f"PCC zero-based index {node} exceeds TSO size")
        if int(bus[node, 1]) != 1:
            raise ValueError(
                f"MATPOWER bus {int(bus[node, 0])} is not a PQ bus; "
                "do not attach a DSO there in this experiment."
            )
    return nodes


def ranked_pcc_candidates(tsppc: dict) -> tuple[int, ...]:
    """Return nonzero-load PQ buses ranked by nominal apparent load."""
    bus = np.asarray(tsppc["bus"], dtype=float)
    candidates = [
        i for i in range(len(bus))
        if int(bus[i, 1]) == 1 and np.hypot(bus[i, 2], bus[i, 3]) > 1e-9
    ]
    candidates.sort(
        key=lambda i: (-float(np.hypot(bus[i, 2], bus[i, 3])), int(bus[i, 0]))
    )
    return tuple(candidates)


def _load_case33_representative_conditions() -> list[dict]:
    """Load the same case33 P10/P30/P50/P70/P90 used previously."""
    selected_path = SOURCE_RESULT_DIR / "selected_dso_conditions.csv"
    raw_path = SOURCE_RESULT_DIR / "coverage_raw_all_50_dso.npz"
    if not selected_path.exists() or not raw_path.exists():
        raise FileNotFoundError(
            "The 50-condition coverage data are missing. Run the existing "
            "case4gs downstream script first, or copy its two coverage files:\n"
            f"  {selected_path}\n  {raw_path}"
        )
    with selected_path.open("r", encoding="utf-8-sig", newline="") as handle:
        selected_rows = list(csv.DictReader(handle))
    raw = np.load(raw_path, allow_pickle=False)
    all_dtheta = np.asarray(raw["dtheta"], dtype=float)
    conditions = []
    for row in selected_rows:
        percentile = int(float(row["coverage_percentile"]))
        scenario_index = int(float(row["dso_scenario_index"]))
        conditions.append({
            "percentile": percentile,
            "scenario_index": scenario_index,
            "coverage_percent": float(row["coverage_percent"]),
            "undercoverage_percent": float(row["undercoverage_percent"]),
            "overcoverage_percent": float(row["overcoverage_percent"]),
            "dtheta": all_dtheta[scenario_index].copy(),
        })
    conditions.sort(key=lambda item: item["percentile"])
    if tuple(item["percentile"] for item in conditions) != FORMAL_PERCENTILES:
        raise RuntimeError("Selected DSO conditions are not P10/P30/P50/P70/P90")
    return conditions


def _load_original_npz_representative_conditions(dim_theta: int) -> list[dict]:
    """Reconstruct large-DS conditions from the requested original data.

    ``approximate_polygon_coverage.py`` generated 50 dtheta vectors with
    RandomState(42) and stored 100 direction-wise coverage ratios per vector.
    The NPZ does not store dtheta itself, so the exact deterministic sequence
    is regenerated and the five empirical percentile cases are selected from
    the per-condition mean PINN coverage.
    """
    path = Path(SOURCE_RESULT_DIR)
    if not path.exists():
        raise FileNotFoundError(f"{DS_CASE} coverage data are missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        n_dtheta = int(data["n_dtheta"])
        n_dirs = int(data["n_dirs"])
        coverage = np.asarray(data["coverage_nn"], dtype=float)
    if coverage.size != n_dtheta * n_dirs:
        raise RuntimeError(
            f"{DS_CASE} coverage size {coverage.size} != {n_dtheta}x{n_dirs}"
        )
    if n_dtheta != legacy.PAPER_N_DTHETA:
        raise RuntimeError(
            f"{DS_CASE} coverage has {n_dtheta} conditions; expected "
            f"{legacy.PAPER_N_DTHETA}"
        )
    ratios = coverage.reshape(n_dtheta, n_dirs)
    dtheta = legacy.generate_paper_dtheta_samples(dim_theta)
    rows = []
    for index in range(n_dtheta):
        stats = legacy.summarize_coverage_ratios(ratios[index])
        rows.append({
            "dso_scenario_index": index,
            **stats,
        })
    selected = legacy.select_coverage_percentile_conditions(rows)
    conditions = [{
        "percentile": int(item["coverage_percentile"]),
        "scenario_index": int(item["dso_scenario_index"]),
        "coverage_percent": float(item["coverage_percent"]),
        "undercoverage_percent": float(item["undercoverage_percent"]),
        "overcoverage_percent": float(item["overcoverage_percent"]),
        "dtheta": dtheta[int(item["dso_scenario_index"])].copy(),
    } for item in selected]
    conditions.sort(key=lambda item: item["percentile"])
    return conditions


def load_representative_conditions(dim_theta: int) -> list[dict]:
    if DS_CASE in {"case118zh_ds", "case533mt_hi_ds", "case36real_3phase_ds"}:
        return _load_original_npz_representative_conditions(dim_theta)
    return _load_case33_representative_conditions()


def prepare_condition_cache(dsppc_nominal: dict, include_formal: bool) -> dict:
    """Load the original model once and calculate reusable DSO condition data."""
    template = legacy.load_nominal_pinn(dsppc_nominal)
    nominal = {
        "name": "nominal",
        "percentile": np.nan,
        "scenario_index": -1,
        "coverage_percent": np.nan,
        "undercoverage_percent": np.nan,
        "overcoverage_percent": np.nan,
        "pinn": template,
        "dsppc": copy.deepcopy(dsppc_nominal),
    }
    nominal["x_ref"] = legacy.nominal_reference_point(
        nominal["dsppc"], nominal["pinn"]
    )
    cache = {"nominal": nominal}
    if not include_formal:
        return cache

    for condition in load_representative_conditions(template["dim_theta"]):
        pinn = legacy.predict_pinn(template, condition["dtheta"])
        dsppc = legacy.apply_dtheta_to_dsppc(dsppc_nominal, pinn)
        item = {
            **condition,
            "name": f'P{condition["percentile"]}',
            "pinn": pinn,
            "dsppc": dsppc,
        }
        item["x_ref"] = legacy.nominal_reference_point(dsppc, pinn)
        cache[item["name"]] = item
    return cache


def make_nominal_host_case(
    tsppc_original: dict,
    dsppc_nominal: dict,
    nominal_x_ref: np.ndarray,
    pcc_nodes: tuple[int, ...],
) -> dict:
    """Decompose each PCC host load once, using the nominal DSO reference."""
    host = copy.deepcopy(tsppc_original)
    ds_base = float(dsppc_nominal["baseMVA"]) * interface_scale()
    for node in pcc_nodes:
        host["bus"][node, 2] -= float(nominal_x_ref[0]) * ds_base
        host["bus"][node, 3] -= float(nominal_x_ref[1]) * ds_base
    return host


def scale_native_host_load(nominal_host: dict, load_factor: float) -> dict:
    scaled = copy.deepcopy(nominal_host)
    scaled["bus"][:, 2] *= float(load_factor)
    scaled["bus"][:, 3] *= float(load_factor)
    return scaled


def _fix_pcc_voltages(model: pyo.ConcreteModel, pcc_nodes: tuple[int, ...]) -> None:
    for node in pcc_nodes:
        model.V[node].fix(PCC_VOLTAGE)


def build_base_model(
    host: dict, assignments: dict[int, dict]
) -> pyo.ConcreteModel:
    pcc_nodes = tuple(assignments)
    model = TD_case.TScase(
        tscasedata=copy.deepcopy(host),
        dscasedata_dict={node: {} for node in pcc_nodes},
    )
    ts_base = float(host["baseMVA"])
    model.multi_dso_base_constraints = pyo.ConstraintList()
    for node, item in assignments.items():
        ds_base = float(item["dsppc"]["baseMVA"]) * interface_scale()
        pd_host = float(host["bus"][node, 2]) / ts_base
        qd_host = float(host["bus"][node, 3]) / ts_base
        model.multi_dso_base_constraints.add(
            model.Pd[node] == pd_host + float(item["x_ref"][0]) * ds_base / ts_base
        )
        model.multi_dso_base_constraints.add(
            model.Qd[node] == qd_host + float(item["x_ref"][1]) * ds_base / ts_base
        )
    _fix_pcc_voltages(model, pcc_nodes)
    return model


def build_pinn_model(
    host: dict, assignments: dict[int, dict]
) -> pyo.ConcreteModel:
    apx = {}
    for node, item in assignments.items():
        apx[node] = {
            "baseMVA": float(item["dsppc"]["baseMVA"]),
            "interface_scale": interface_scale(),
            "A_hat": np.asarray(item["pinn"]["A"], dtype=float),
            "b_hat": np.asarray(item["pinn"]["b"], dtype=float),
        }
    model = TD_case.TDcase(
        tscasedata=copy.deepcopy(host), dscasedata_dict=apx, is_apx=True
    )
    _fix_pcc_voltages(model, tuple(assignments))
    return model


def build_true_model(
    host: dict, assignments: dict[int, dict]
) -> pyo.ConcreteModel:
    ds_cases = {}
    for node, item in assignments.items():
        ds_case = copy.deepcopy(item["dsppc"])
        ds_case["interface_scale"] = interface_scale()
        ds_cases[node] = ds_case
    model = TD_case.TDcase(
        tscasedata=copy.deepcopy(host), dscasedata_dict=ds_cases,
        is_apx=False, is_3phase=is_three_phase(),
    )
    _fix_pcc_voltages(model, tuple(assignments))
    return model


def interface_points(
    model: pyo.ConcreteModel,
    region: str,
    assignments: dict[int, dict],
) -> np.ndarray:
    values = []
    for node, item in assignments.items():
        if region == "base":
            point = np.asarray(item["x_ref"], dtype=float)
        else:
            block = model.DS[node]
            if hasattr(block, "var_proj"):
                point = np.array(
                    [pyo.value(block.var_proj[0]),
                     pyo.value(block.var_proj[1])], dtype=float,
                )
            else:
                # The legacy single-phase polygon block exposes its interface
                # through Pn/Qn; full physical and three-phase blocks expose
                # the equivalent aggregate variable var_proj.
                point = np.array(
                    [pyo.value(block.Pn[1]), pyo.value(block.Qn[1])],
                    dtype=float,
                )
        values.append(point)
    return np.asarray(values, dtype=float)


def make_assignments(
    pcc_nodes: tuple[int, ...], cache: dict, profile_index: int | None
) -> dict[int, dict]:
    if profile_index is None:
        return {node: cache["nominal"] for node in pcc_nodes}
    names = [f"P{value}" for value in FORMAL_PERCENTILES]
    # Each profile is dominated by one target coverage percentile but remains
    # heterogeneous: per group of five DSOs, three use the target condition and
    # one uses each adjacent condition (edge profiles become 4:1).  Therefore
    # aggregate coverage changes from profile to profile instead of merely
    # permuting the same five conditions among PCC buses.
    offsets = (0, 0, 0, -1, 1)
    return {
        node: cache[names[int(np.clip(
            profile_index + offsets[position % len(offsets)],
            0,
            len(names) - 1,
        ))]]
        for position, node in enumerate(pcc_nodes)
    }


def solve_three_regions(host: dict, assignments: dict[int, dict]) -> dict:
    builders = {
        "base": build_base_model,
        "pinn": build_pinn_model,
        "true": build_true_model,
    }
    solved = {}
    for region, builder in builders.items():
        build_start = time.perf_counter()
        model = builder(host, assignments)
        build_time = time.perf_counter() - build_start
        if region == "true":
            # The three-phase blocks converge to a physically better AC branch
            # from the near-nominal 0.99-p.u./0.30-flow initialization.  The
            # original 0.96-p.u. initialization is retained for single-phase
            # cases to preserve the completed V1--V3 timing protocol.
            legacy.initialize_true_model(model, 1 if is_three_phase() else 0)
        result = legacy.solve_model(model)
        points = (
            interface_points(model, region, assignments)
            if result["feasible_solution"]
            else np.full((len(assignments), 2), np.nan)
        )
        solved[region] = {
            "model": model,
            "result": result,
            "points": points,
            "build_time_s": build_time,
        }
    return solved


def screen_pcc_nodes(
    tsppc: dict,
    dsppc: dict,
    cache: dict,
    target_count: int,
) -> tuple[int, ...]:
    """Greedily select a feasible fixed-voltage PCC set using Base only.

    A candidate is accepted only when the complete prefix remains an economic
    optimum.  Consequently every prefix of the saved set (e.g. 5 and 10 DSOs)
    has already passed the same Base feasibility test used in the comparison.
    """
    candidates = ranked_pcc_candidates(tsppc)
    if target_count > len(candidates):
        raise ValueError(
            f"Requested {target_count} PCCs but only {len(candidates)} "
            "nonzero-load PQ buses are available."
        )

    selected: list[int] = []
    rows: list[dict] = []
    bus = np.asarray(tsppc["bus"], dtype=float)
    print(
        f"\n[PCC screen] target={target_count}; candidates={len(candidates)}; "
        f"fixed voltage={PCC_VOLTAGE:.3f} p.u."
    )
    for rank, node in enumerate(candidates, start=1):
        trial_nodes = tuple([*selected, node])
        host = make_nominal_host_case(
            tsppc, dsppc, cache["nominal"]["x_ref"], trial_nodes
        )
        assignments = make_assignments(trial_nodes, cache, profile_index=None)
        build_start = time.perf_counter()
        model = build_base_model(host, assignments)
        build_time = time.perf_counter() - build_start
        result = legacy.solve_model(model)
        accepted = bool(result["optimization_success"])
        if accepted:
            selected.append(node)
        rows.append({
            "candidate_rank": rank,
            "pcc_index_zero_based": node,
            "pcc_bus_number": int(bus[node, 0]),
            "nominal_p_mw": float(bus[node, 2]),
            "nominal_q_mvar": float(bus[node, 3]),
            "nominal_apparent_load_mva": float(
                np.hypot(bus[node, 2], bus[node, 3])
            ),
            "trial_dso_count": len(trial_nodes),
            "accepted": accepted,
            "selected_order": len(selected) if accepted else np.nan,
            "feasible_solution": result["feasible_solution"],
            "optimization_success": result["optimization_success"],
            "termination": result["termination"],
            "objective": result["objective"],
            "build_time_s": build_time,
            "solve_time_s": result["solve_time"],
            "pcc_voltage_pu": PCC_VOLTAGE,
            "ts_case": TS_CASE,
            "ds_case": DS_CASE,
            "ts_model_version": TS_MODEL_VERSION,
        })
        print(
            f"  bus={int(bus[node, 0]):3d}, "
            f"|S_load|={np.hypot(bus[node, 2], bus[node, 3]):7.3f} MVA, "
            f"accepted={accepted}, selected={len(selected)}/{target_count}, "
            f"termination={result['termination']}"
        )
        _write_csv(OUT_ROOT / PCC_SELECTION_FILE, rows)
        if len(selected) == target_count:
            break

    if len(selected) < target_count:
        raise RuntimeError(
            f"Only {len(selected)} mutually feasible PCCs were found; "
            f"see {OUT_ROOT / PCC_SELECTION_FILE}."
        )
    print(
        "[PCC screen] selected MATPOWER buses: "
        + ", ".join(str(int(bus[node, 0])) for node in selected)
    )
    return tuple(selected)


def load_screened_pcc_nodes(tsppc: dict, target_count: int) -> tuple[int, ...]:
    """Return the immutable PCC set belonging to the active version."""
    config = VERSION_CONFIGS[EXPERIMENT_VERSION]
    configured_bus_numbers = tuple(int(value) for value in config["pcc_bus_numbers"])
    if target_count > len(configured_bus_numbers):
        raise ValueError(
            f"{EXPERIMENT_VERSION} defines only {len(configured_bus_numbers)} "
            f"PCCs, but {target_count} were requested"
        )
    bus = np.asarray(tsppc["bus"], dtype=float)
    number_to_index = {int(row[0]): index for index, row in enumerate(bus)}
    missing = [number for number in configured_bus_numbers if number not in number_to_index]
    if missing:
        raise RuntimeError(f"Configured MATPOWER buses are missing: {missing}")
    nodes = tuple(number_to_index[number] for number in configured_bus_numbers)
    return validate_pcc_nodes(tsppc, target_count, nodes)


def summarize_scenario(
    solved: dict,
    assignments: dict[int, dict],
    n_dso: int,
    profile_index: int | None,
    ts_load_factor: float,
    ts_base_mva: float,
    run_disaggregation: bool = True,
) -> tuple[dict, list[dict]]:
    results = {key: value["result"] for key, value in solved.items()}
    all_optimal = all(item["optimization_success"] for item in results.values())
    costs = {key: float(item["objective"]) for key, item in results.items()}
    if all_optimal:
        order_ok = bool(
            costs["true"] <= costs["pinn"] + 1e-6
            and costs["pinn"] <= costs["base"] + 1e-6
        )
        true_value = costs["base"] - costs["true"]
        pinn_value = costs["base"] - costs["pinn"]
        retention = _safe_percent(pinn_value, true_value)
        relative_gap = _safe_percent(costs["pinn"] - costs["true"], costs["true"])
    else:
        order_ok = False
        true_value = pinn_value = retention = relative_gap = np.nan

    pinn_points = solved["pinn"]["points"]
    true_points = solved["true"]["points"]
    point_delta = np.abs(pinn_points - true_points) * interface_scale()
    dispatch_delta = np.abs(
        results["pinn"]["dispatch_p_pu"] - results["true"]["dispatch_p_pu"]
    )
    ts_base = float(ts_base_mva)

    dso_rows = []
    disaggregation_errors = []
    disaggregation_success = []
    for position, (node, item) in enumerate(assignments.items()):
        if not run_disaggregation:
            continue
        if results["pinn"]["optimization_success"]:
            check = legacy.disaggregation_check(pinn_points[position], item["dsppc"])
        else:
            check = {
                "squared_error": np.nan,
                "feasible_solution": False,
                "optimization_success": False,
                "termination": "PINN dispatch did not converge",
                "solve_time_s": np.nan,
            }
        disaggregation_errors.append(float(check["squared_error"]))
        disaggregation_success.append(bool(check["optimization_success"]))
        dso_rows.append({
            "n_dso": n_dso,
            "profile": (
                "nominal" if profile_index is None
                else f"target_P{FORMAL_PERCENTILES[profile_index]}"
            ),
            "ts_load_factor": ts_load_factor,
            "dso_position": position + 1,
            "pcc_index_zero_based": node,
            "pcc_bus_number": int(node + 1),
            "condition": item["name"],
            "coverage_percent": item["coverage_percent"],
            "undercoverage_percent": item["undercoverage_percent"],
            "overcoverage_percent": item["overcoverage_percent"],
            "pinn_p_pu_dsbase": pinn_points[position, 0],
            "pinn_q_pu_dsbase": pinn_points[position, 1],
            "true_p_pu_dsbase": true_points[position, 0],
            "true_q_pu_dsbase": true_points[position, 1],
            "pinn_true_interface_distance_pu": float(
                np.linalg.norm(pinn_points[position] - true_points[position])
            ),
            "disaggregation_squared_error": check["squared_error"],
            "disaggregation_feasible": check["feasible_solution"],
            "disaggregation_optimization_success": check["optimization_success"],
            "disaggregation_termination": check["termination"],
            "disaggregation_solve_time_s": check["solve_time_s"],
        })

    finite_disagg = np.asarray(disaggregation_errors, dtype=float)
    finite_disagg = finite_disagg[np.isfinite(finite_disagg)]
    condition_coverages = np.asarray(
        [item["coverage_percent"] for item in assignments.values()], dtype=float
    )
    condition_undercoverage = np.asarray(
        [item["undercoverage_percent"] for item in assignments.values()], dtype=float
    )
    condition_overcoverage = np.asarray(
        [item["overcoverage_percent"] for item in assignments.values()], dtype=float
    )
    profile_name = (
        "nominal" if profile_index is None
        else f"target_P{FORMAL_PERCENTILES[profile_index]}"
    )
    row = {
        "n_dso": n_dso,
        "profile": profile_name,
        "ts_load_factor": ts_load_factor,
        "interface_divisor": interface_divisor(),
        "interface_scale": interface_scale(),
        "mean_dso_coverage_percent": (
            float(np.nanmean(condition_coverages))
            if np.any(np.isfinite(condition_coverages)) else np.nan
        ),
        "mean_dso_undercoverage_percent": (
            float(np.nanmean(condition_undercoverage))
            if np.any(np.isfinite(condition_undercoverage)) else np.nan
        ),
        "mean_dso_overcoverage_percent": (
            float(np.nanmean(condition_overcoverage))
            if np.any(np.isfinite(condition_overcoverage)) else np.nan
        ),
        "base_feasible": results["base"]["feasible_solution"],
        "pinn_feasible": results["pinn"]["feasible_solution"],
        "true_feasible": results["true"]["feasible_solution"],
        "base_optimal": results["base"]["optimization_success"],
        "pinn_optimal": results["pinn"]["optimization_success"],
        "true_optimal": results["true"]["optimization_success"],
        "all_optimal": all_optimal,
        "base_termination": results["base"]["termination"],
        "pinn_termination": results["pinn"]["termination"],
        "true_termination": results["true"]["termination"],
        "base_cost": costs["base"],
        "pinn_cost": costs["pinn"],
        "true_cost": costs["true"],
        "cost_order_ok": order_ok,
        "pinn_true_cost_gap": costs["pinn"] - costs["true"] if all_optimal else np.nan,
        "pinn_true_relative_cost_gap_percent": relative_gap,
        "true_flexibility_value": true_value,
        "pinn_flexibility_value": pinn_value,
        "value_retention_percent": retention,
        "pinn_true_pg_l1_mw": float(np.sum(dispatch_delta) * ts_base),
        "pinn_true_pg_linf_mw": float(np.max(dispatch_delta) * ts_base),
        "pinn_true_interface_l1_pu": float(np.sum(point_delta)),
        "pinn_true_interface_linf_pu": float(np.max(point_delta)),
        "base_build_time_s": solved["base"]["build_time_s"],
        "pinn_build_time_s": solved["pinn"]["build_time_s"],
        "true_build_time_s": solved["true"]["build_time_s"],
        "base_solve_time_s": results["base"]["solve_time"],
        "pinn_solve_time_s": results["pinn"]["solve_time"],
        "true_solve_time_s": results["true"]["solve_time"],
        "disaggregation_success_ratio_percent": (
            100.0 * float(np.mean(disaggregation_success))
            if disaggregation_success else np.nan
        ),
        "disaggregation_mean_squared_error": (
            float(np.mean(finite_disagg)) if finite_disagg.size else np.nan
        ),
        "disaggregation_p95_squared_error": (
            float(np.percentile(finite_disagg, 95)) if finite_disagg.size else np.nan
        ),
        "disaggregation_max_squared_error": (
            float(np.max(finite_disagg)) if finite_disagg.size else np.nan
        ),
        "disaggregation_all_below_tolerance": bool(
            finite_disagg.size == n_dso
            and np.all(finite_disagg <= DISAGGREGATION_TOL)
        ),
    }
    return row, dso_rows


def run_pilot() -> None:
    out = OUT_ROOT / "pilot"
    tsppc = getattr(TD_case, TS_CASE)()
    dsppc = load_ds_case()
    cache = prepare_condition_cache(dsppc, include_formal=False)
    screened_nodes = load_screened_pcc_nodes(tsppc, max(PILOT_N_DSO))
    rows, dso_rows = [], []
    for n_dso in PILOT_N_DSO:
        nodes = validate_pcc_nodes(tsppc, n_dso, screened_nodes)
        host_nominal = make_nominal_host_case(
            tsppc, dsppc, cache["nominal"]["x_ref"], nodes
        )
        assignments = make_assignments(nodes, cache, profile_index=None)
        print(f"\n[pilot] case118 + {n_dso} {DS_CASE}, TSO load=1.0")
        solved = solve_three_regions(host_nominal, assignments)
        row, details = summarize_scenario(
            solved, assignments, n_dso, None, 1.0, tsppc["baseMVA"]
        )
        rows.append(row)
        dso_rows.extend(details)
        print(
            f"  optimal Base/PINN/True="
            f"{row['base_optimal']}/{row['pinn_optimal']}/{row['true_optimal']}; "
            f"cost order={row['cost_order_ok']}; retention={row['value_retention_percent']:.3f}%"
        )
    _write_csv(out / "pilot_summary.csv", rows)
    _write_csv(out / "pilot_dso_details.csv", dso_rows)
    print(f"\nPilot results: {out}")


def run_formal(n_dso: int, timing_only: bool = False) -> None:
    out = OUT_ROOT / _formal_directory_name(n_dso, timing_only)
    tsppc = getattr(TD_case, TS_CASE)()
    nodes = load_screened_pcc_nodes(tsppc, n_dso)
    dsppc = load_ds_case()
    cache = prepare_condition_cache(dsppc, include_formal=True)
    host_nominal = make_nominal_host_case(
        tsppc, dsppc, cache["nominal"]["x_ref"], nodes
    )

    rows, dso_rows = [], []
    total = len(FORMAL_PERCENTILES) * len(FORMAL_TS_LOAD_FACTORS)
    count = 0
    for profile_index in range(len(FORMAL_PERCENTILES)):
        assignments = make_assignments(nodes, cache, profile_index)
        condition_names = ",".join(item["name"] for item in assignments.values())
        for load_factor in FORMAL_TS_LOAD_FACTORS:
            count += 1
            print(
                f"\n[formal {count}/{total}] nDSO={n_dso}, "
                f"profile={profile_index + 1}, TS load={load_factor:.2f}\n"
                f"  DSO conditions: {condition_names}"
            )
            host = scale_native_host_load(host_nominal, load_factor)
            solved = solve_three_regions(host, assignments)
            row, details = summarize_scenario(
                solved, assignments, n_dso, profile_index, load_factor,
                tsppc["baseMVA"],
                run_disaggregation=not timing_only,
            )
            rows.append(row)
            dso_rows.extend(details)
            # Incremental saving protects a long server run from losing all data.
            _write_csv(out / "formal_summary.csv", rows)
            _write_csv(out / "formal_dso_details.csv", dso_rows)
            _write_solver_time_statistics(
                out / "solver_time_statistics.csv", rows
            )
            print(
                f"  all optimal={row['all_optimal']}, cost order={row['cost_order_ok']}, "
                f"gap={row['pinn_true_relative_cost_gap_percent']:.5f}%, "
                f"retention={row['value_retention_percent']:.3f}%"
            )

    if timing_only:
        print(f"\nSolver-time results: {out}")
        return

    valid = [row for row in rows if row["all_optimal"]]
    evidence = [{
        "n_scenarios": len(rows),
        "all_optimal_scenarios": len(valid),
        "cost_order_success_ratio_percent": 100.0 * float(
            np.mean([row["cost_order_ok"] for row in rows])
        ),
        "mean_relative_cost_gap_percent": float(np.nanmean(
            [row["pinn_true_relative_cost_gap_percent"] for row in rows]
        )),
        "max_relative_cost_gap_percent": float(np.nanmax(
            [row["pinn_true_relative_cost_gap_percent"] for row in rows]
        )),
        "mean_value_retention_percent": float(np.nanmean(
            [row["value_retention_percent"] for row in rows]
        )),
        "min_value_retention_percent": float(np.nanmin(
            [row["value_retention_percent"] for row in rows]
        )),
        "mean_pg_l1_deviation_mw": float(np.nanmean(
            [row["pinn_true_pg_l1_mw"] for row in rows]
        )),
        "max_pg_linf_deviation_mw": float(np.nanmax(
            [row["pinn_true_pg_linf_mw"] for row in rows]
        )),
        "disaggregation_success_ratio_percent": 100.0 * float(np.mean(
            [detail["disaggregation_optimization_success"] for detail in dso_rows]
        )),
        "max_disaggregation_squared_error": float(np.nanmax(
            [detail["disaggregation_squared_error"] for detail in dso_rows]
        )),
    }]
    _write_csv(out / "reviewer_evidence_summary.csv", evidence)
    _write_solver_time_statistics(out / "solver_time_statistics.csv", rows)
    print(f"\nFormal results: {out}")


def run_representative_multistart(n_dso: int) -> dict:
    """Audit local-solution sensitivity using identical start rules.

    The three model classes do not contain exactly the same variables, but
    ``legacy.initialize_true_model`` assigns the same voltage, angle,
    generator, and flow initialization rule to every common variable.  The
    True-only DSO variables receive the corresponding deterministic start.
    Formal single-run results are not overwritten by this diagnostic.
    """
    out = OUT_ROOT / _formal_directory_name(n_dso, timing_only=False)
    tsppc = getattr(TD_case, TS_CASE)()
    nodes = load_screened_pcc_nodes(tsppc, n_dso)
    dsppc = load_ds_case()
    cache = prepare_condition_cache(dsppc, include_formal=True)
    host_nominal = make_nominal_host_case(
        tsppc, dsppc, cache["nominal"]["x_ref"], nodes
    )

    builders = {
        "base": build_base_model,
        "pinn": build_pinn_model,
        "true": build_true_model,
    }
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    order_rows: list[dict] = []

    total = len(REPRESENTATIVE_MULTISTART) * len(builders) * MULTISTART_COUNT
    counter = 0
    print("\n" + "=" * 76)
    print(
        "Representative multi-start audit: "
        f"{len(REPRESENTATIVE_MULTISTART)} scenarios x "
        f"{len(builders)} models x {MULTISTART_COUNT} starts = {total} solves"
    )
    print("=" * 76)

    for profile_index, load_factor in REPRESENTATIVE_MULTISTART:
        profile_name = f"target_P{FORMAL_PERCENTILES[profile_index]}"
        assignments = make_assignments(nodes, cache, profile_index)
        host = scale_native_host_load(host_nominal, load_factor)
        scenario_rows: list[dict] = []
        print(f"\n[multistart] profile={profile_name}, TS load={load_factor:.2f}")

        for region, builder in builders.items():
            for start_index in range(MULTISTART_COUNT):
                counter += 1
                build_start = time.perf_counter()
                model = builder(host, assignments)
                build_time = time.perf_counter() - build_start
                # Apply the same deterministic start rule to all three models.
                legacy.initialize_true_model(model, start_index)
                result = legacy.solve_model(model)
                row = {
                    "profile": profile_name,
                    "profile_index": profile_index,
                    "ts_load_factor": load_factor,
                    "region": region,
                    "start_index": start_index,
                    "feasible_solution": result["feasible_solution"],
                    "optimization_success": result["optimization_success"],
                    "termination": result["termination"],
                    "cost": result["objective"],
                    "candidate_cost": result["candidate_objective"],
                    "build_time_s": build_time,
                    "solve_time_s": result["solve_time"],
                }
                raw_rows.append(row)
                scenario_rows.append(row)
                _write_csv(out / "representative_multistart_raw.csv", raw_rows)
                print(
                    f"  {counter:02d}/{total} {region:4s} start={start_index}: "
                    f"optimal={row['optimization_success']} "
                    f"termination={row['termination']} cost={row['cost']}"
                )

        best_costs: dict[str, float] = {}
        for region in builders:
            region_rows = [
                row for row in scenario_rows
                if row["region"] == region and row["optimization_success"]
                and np.isfinite(row["cost"])
            ]
            costs = np.asarray([row["cost"] for row in region_rows], dtype=float)
            if costs.size:
                best_position = int(np.argmin(costs))
                best_cost = float(costs[best_position])
                worst_cost = float(np.max(costs))
                spread = _safe_percent(worst_cost - best_cost, abs(best_cost))
                best_start = int(region_rows[best_position]["start_index"])
                best_costs[region] = best_cost
            else:
                best_cost = worst_cost = spread = np.nan
                best_start = -1
            summary_rows.append({
                "profile": profile_name,
                "ts_load_factor": load_factor,
                "region": region,
                "n_starts": MULTISTART_COUNT,
                "n_feasible": sum(
                    bool(row["feasible_solution"])
                    for row in scenario_rows if row["region"] == region
                ),
                "n_optimal": int(costs.size),
                "best_start_index": best_start,
                "best_cost": best_cost,
                "worst_cost": worst_cost,
                "mean_cost": float(np.mean(costs)) if costs.size else np.nan,
                "std_cost": float(np.std(costs)) if costs.size else np.nan,
                "relative_cost_spread_percent": spread,
                "median_solve_time_s": float(np.median([
                    row["solve_time_s"] for row in region_rows
                ])) if region_rows else np.nan,
            })

        all_costs = all(region in best_costs for region in builders)
        cost_order_ok = bool(
            all_costs
            and best_costs["true"] <= best_costs["pinn"] + 1e-6
            and best_costs["pinn"] <= best_costs["base"] + 1e-6
        )
        order_rows.append({
            "profile": profile_name,
            "ts_load_factor": load_factor,
            "all_models_have_optimal_start": all_costs,
            "best_base_cost": best_costs.get("base", np.nan),
            "best_pinn_cost": best_costs.get("pinn", np.nan),
            "best_true_cost": best_costs.get("true", np.nan),
            "best_cost_order_ok": cost_order_ok,
            "best_pinn_true_cost_gap": (
                best_costs["pinn"] - best_costs["true"]
                if all_costs else np.nan
            ),
            "best_pinn_true_relative_gap_percent": (
                _safe_percent(
                    best_costs["pinn"] - best_costs["true"],
                    best_costs["true"],
                ) if all_costs else np.nan
            ),
        })
        _write_csv(out / "representative_multistart_summary.csv", summary_rows)
        _write_csv(out / "representative_multistart_cost_order.csv", order_rows)

    finite_spreads = np.asarray([
        row["relative_cost_spread_percent"] for row in summary_rows
    ], dtype=float)
    finite_spreads = finite_spreads[np.isfinite(finite_spreads)]
    audit = {
        "multistart_scenarios": len(REPRESENTATIVE_MULTISTART),
        "multistart_count_per_model": MULTISTART_COUNT,
        "multistart_total_solves": len(raw_rows),
        "multistart_optimal_solve_ratio_percent": 100.0 * float(np.mean([
            bool(row["optimization_success"]) for row in raw_rows
        ])),
        "multistart_cost_order_success_ratio_percent": 100.0 * float(np.mean([
            bool(row["best_cost_order_ok"]) for row in order_rows
        ])),
        "multistart_max_relative_cost_spread_percent": (
            float(np.max(finite_spreads)) if finite_spreads.size else np.nan
        ),
    }
    _write_csv(out / "representative_multistart_evidence.csv", [audit])

    # When the formal run is available, append the audit to the one-row
    # reviewer evidence file as well.  The detailed formal and multi-start
    # CSV files remain separate, so this convenience merge does not change
    # any previously defined metric.
    evidence_path = out / "reviewer_evidence_summary.csv"
    if evidence_path.exists():
        with evidence_path.open("r", newline="", encoding="utf-8-sig") as handle:
            formal_rows = list(csv.DictReader(handle))
        if formal_rows:
            formal_rows[0].update(audit)
            _write_csv(evidence_path, formal_rows)

    print(f"\nMulti-start results: {out}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", choices=tuple(VERSION_CONFIGS) + ("new", "all"),
        default=DEFAULT_VERSION,
        help=(
            "v1=case33, v2=case533, v3=case118, v4=case36 three-phase; "
            "each version contains 27 DSOs."
        ),
    )
    parser.add_argument(
        "--mode", choices=("timing", "all", "pilot", "formal", "multistart"),
        default=DEFAULT_MODE,
        help=(
            "timing runs only the formal profile scenarios and is the direct-"
            "execution default; formal additionally runs the multistart audit."
        ),
    )
    parser.add_argument(
        "--n-dso", type=int, default=None,
        help="Optional consistency check; each version has a fixed DSO count.",
    )
    return parser.parse_args()


def _run_one_version(version: str, mode: str, requested_n_dso: int | None,
                     ipopt: dict[str, str]) -> None:
    config = configure_version(version)
    configured_n_dso = int(config["n_dso"])
    if requested_n_dso is not None and requested_n_dso != configured_n_dso:
        raise ValueError(
            f"{version} is fixed at {configured_n_dso} DSOs; "
            f"received --n-dso {requested_n_dso}"
        )
    print(
        f"R1.5/R3.2 large downstream validation | version={version} "
        f"({config['label']}) | TSO={TS_CASE} | DSO={DS_CASE} x {configured_n_dso} | "
        f"device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )
    print(
        "PCC set: immutable for the selected version; PQ buses only."
    )
    tsppc = getattr(TD_case, TS_CASE)()
    n_rated = int(np.count_nonzero(np.asarray(tsppc["branch"])[:, 5] > 0.0))
    print(
        "PCC voltage is fixed at 1.0 p.u.; DSO thermal limits are not added."
    )
    print(
        f"TS AC model version: {TS_MODEL_VERSION}; "
        f"positive RATE_A branches={n_rated}/{len(tsppc['branch'])}."
    )
    if mode == "all":
        run_pilot()
        config_path = save_experiment_config(ipopt, configured_n_dso)
        print(f"Experiment metadata: {config_path}")
        run_formal(configured_n_dso)
        run_representative_multistart(configured_n_dso)
    elif mode == "pilot":
        run_pilot()
    elif mode in ("timing", "formal"):
        config_path = save_experiment_config(
            ipopt, configured_n_dso, timing_only=(mode == "timing")
        )
        print(f"Experiment metadata: {config_path}")
        run_formal(configured_n_dso, timing_only=(mode == "timing"))
        if mode == "formal":
            run_representative_multistart(configured_n_dso)
    else:
        config_path = save_experiment_config(ipopt, configured_n_dso)
        print(f"Experiment metadata: {config_path}")
        run_representative_multistart(configured_n_dso)


def _write_cross_version_timing_summary(versions: tuple[str, ...]) -> None:
    statistics, speedups = [], []
    for version in versions:
        config = VERSION_CONFIGS[version]
        timing_dir = (
            VERSION_OUT_ROOT / str(config["label"])
            / _formal_directory_name(int(config["n_dso"]), timing_only=True)
        )
        formal_dir = (
            VERSION_OUT_ROOT / str(config["label"])
            / _formal_directory_name(int(config["n_dso"]), timing_only=False)
        )
        source_dir = timing_dir if timing_dir.exists() else formal_dir
        for filename, destination in (
            ("solver_time_statistics.csv", statistics),
            ("solver_time_speedup.csv", speedups),
        ):
            path = source_dir / filename
            if path.exists():
                with path.open("r", newline="", encoding="utf-8-sig") as handle:
                    destination.extend(csv.DictReader(handle))
    _write_csv(VERSION_OUT_ROOT / "solver_time_comparison.csv", statistics)
    _write_csv(VERSION_OUT_ROOT / "solver_time_speedup_comparison.csv", speedups)


def main() -> None:
    args = parse_args()
    if args.version in {"all", "new"} and args.mode != "timing":
        raise ValueError(
            "--version all/new is reserved for --mode timing; select one version "
            "for pilot, formal, multistart, or all."
        )
    if args.version in {"all", "new"} and args.n_dso is not None:
        raise ValueError("--n-dso cannot be combined with --version all/new.")

    ipopt = prepare_ipopt()
    print(f"Ipopt: {ipopt['executable']}")
    print(f"Ipopt version: {ipopt['version']}")
    if args.version == "all":
        versions = tuple(VERSION_CONFIGS)
    elif args.version == "new":
        versions = ("v3", "v4")
    else:
        versions = (args.version,)
    for index, version in enumerate(versions, start=1):
        if len(versions) > 1:
            print(f"\n{'=' * 78}\nSolver-time experiment version {index}/{len(versions)}: {version}")
        _run_one_version(version, args.mode, args.n_dso, ipopt)
    # Include earlier completed versions in the combined table even when the
    # PyCharm default runs only the newly added V4.
    _write_cross_version_timing_summary(tuple(VERSION_CONFIGS))
    print(f"\nSolver-time summary: {VERSION_OUT_ROOT / 'solver_time_comparison.csv'}")
    print(
        "Speedup summary: "
        f"{VERSION_OUT_ROOT / 'solver_time_speedup_comparison.csv'}"
    )


if __name__ == "__main__":
    main()
