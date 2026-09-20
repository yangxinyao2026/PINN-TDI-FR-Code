# -*- coding: utf-8 -*-
"""Run the R1.5/R3.2 small-system downstream transmission-distribution dispatch experiment."""

import argparse
import contextlib
import copy
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import torch

# Compatible ``python -m`` withPyCharmRun this file directly.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from Simulator import PROJECT_ROOT
from Simulator.Approximator import FullNet, PreTrainNet
from Simulator.cases import TD_case
from Simulator.draw_pictures.approximate_polygon_coverage import (
    compute_k_max_polytope,
    compute_k_max_ray,
    create_ray_model,
    generate_random_directions,
    update_ray_model_params,
)
from Simulator.reference_point import (
    REFERENCE_MEMBERSHIP_TOL,
    ReferencePointCalculator,
    reference_point_membership,
)
from Simulator.solver_environment import prepare_ipopt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TS_CASE = 'case4gs_ts'
DS_CASE = 'case33bw_ds'
PCC_NODE = 1                 # TScaseof0base index: case4gsbusbar2 (PQload node)
PCC_VOLTAGE = 1.0
SOLVER_NAME = 'ipopt'
TEE = False
DIAGNOSTIC_CURRENT_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0)
DIAGNOSTIC_CURRENT_MULTIPLIER = DIAGNOSTIC_CURRENT_MULTIPLIERS[0]
DIAGNOSTIC_CURRENT_FLOOR_PU = 0.05
TS_LOAD_FACTORS = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
GRID_TS_LOAD_FACTORS = (0.80, 0.90, 1.00, 1.10, 1.20)
GRID_COVERAGE_PERCENTILES = (10, 30, 50, 70, 90)
PAPER_DTHETA_SEED = 42
PAPER_DTHETA_RANGE = (-0.5, 0.5)
PAPER_N_DTHETA = 50
PAPER_N_DIRS = 100
GRID_OUT_DIRNAME = 'operating_grid_random_dtheta'
TS_LOAD_FACTOR = 1.0
USE_DSO_CURRENT_LIMITS = False
COVERAGE_N_DIRS = 720
TRUE_MULTISTART_COUNT = 5

PAPER_MODEL_DIR = (
    PROJECT_ROOT / 'results' / 'ds_proj_paper' / DS_CASE
    / 'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)'
)
PRETRAIN_WEIGHTS = PAPER_MODEL_DIR / 'pretrainnet_weights.pth'
FULLNET_WEIGHTS = PAPER_MODEL_DIR / 'fullnet_weights_feasible.pth'
OUT_ROOT = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'downstream' / f'{TS_CASE}_{DS_CASE}'
)
OUT = OUT_ROOT


def _termination_text(result):
    return str(result.solver.termination_condition)


def _has_feasible_solution(result):
    acceptable = {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.feasible,
    }
    return result.solver.termination_condition in acceptable


def _is_optimization_success(result):
    """Only these statuses justify reporting an economic optimum."""
    acceptable = {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
    }
    return result.solver.termination_condition in acceptable


def _is_success(result):
    """Backward-compatible strict success used by optimization diagnostics."""
    return _is_optimization_success(result)


def summarize_coverage_ratios(coverage):
    """Summaryapproximate_polygon_coverage.pyDefined radial ratio.

    Keep it as an internal function of this script to avoid requiring users to modify the original coverage script simultaneously..
    """
    coverage = np.asarray(coverage, dtype=float)
    coverage = coverage[np.isfinite(coverage)]
    if coverage.size == 0:
        raise ValueError('No valid radialcoveragesample')
    under = np.maximum(1.0 - coverage, 0.0)
    over = np.maximum(coverage - 1.0, 0.0)
    tol = 1e-8
    return {
        'coverage_percent': 100.0 * float(np.mean(coverage)),
        'undercoverage_percent': 100.0 * float(np.mean(under)),
        'overcoverage_percent': 100.0 * float(np.mean(over)),
        'undercovered_direction_percent': 100.0 * float(
            np.mean(coverage < 1.0 - tol)),
        'overcovered_direction_percent': 100.0 * float(
            np.mean(coverage > 1.0 + tol)),
        'coverage_median_percent': 100.0 * float(np.median(coverage)),
        'coverage_p05_percent': 100.0 * float(np.percentile(coverage, 5)),
        'coverage_p95_percent': 100.0 * float(np.percentile(coverage, 95)),
        'coverage_sample_count': int(coverage.size),
    }


def solve_model(model):
    solver = pyo.SolverFactory(SOLVER_NAME)
    if not solver.available(False):
        raise RuntimeError(f'Solver is not available: {SOLVER_NAME}')
    start = time.perf_counter()
    result = solver.solve(model, tee=TEE)
    elapsed = time.perf_counter() - start
    feasible_solution = _has_feasible_solution(result)
    optimization_success = _is_optimization_success(result)
    candidate_objective = (
        float(pyo.value(model.obj)) if feasible_solution else np.nan
    )
    # Cost comparisons only use proven local/global optima. A merely feasible
    # termination is retained separately as a diagnostic candidate.
    objective = candidate_objective if optimization_success else np.nan
    if feasible_solution:
        dispatch_p = np.array(
            [pyo.value(model.Pg[g], exception=False) for g in model.G], dtype=float
        )
        dispatch_q = np.array(
            [pyo.value(model.Qg[g], exception=False) for g in model.G], dtype=float
        )
    else:
        dispatch_p = np.full(len(model.G), np.nan, dtype=float)
        dispatch_q = np.full(len(model.G), np.nan, dtype=float)
    branch_apparent_from_pu = np.full(len(model.L), np.nan, dtype=float)
    branch_apparent_to_pu = np.full(len(model.L), np.nan, dtype=float)
    branch_thermal_utilization = np.full(len(model.L), np.nan, dtype=float)
    if feasible_solution and hasattr(model, 'ts_rate_a_pu'):
        for l in model.L:
            index = int(l)
            p_from = float(pyo.value(model.ts_p_from_pu[l]))
            q_from = float(pyo.value(model.ts_q_from_pu[l]))
            p_to = float(pyo.value(model.ts_p_to_pu[l]))
            q_to = float(pyo.value(model.ts_q_to_pu[l]))
            branch_apparent_from_pu[index] = np.hypot(p_from, q_from)
            branch_apparent_to_pu[index] = np.hypot(p_to, q_to)
            rating = float(pyo.value(model.ts_rate_a_pu[l]))
            if rating > 0.0:
                branch_thermal_utilization[index] = (
                    max(branch_apparent_from_pu[index],
                        branch_apparent_to_pu[index]) / rating
                )
    rated_utilization = branch_thermal_utilization[
        np.isfinite(branch_thermal_utilization)
    ]
    return {
        'result': result,
        # ``success`` remains an alias for strict economic-optimization success
        # so old readers cannot accidentally treat a feasible point as optimum.
        'success': optimization_success,
        'feasible_solution': feasible_solution,
        'optimization_success': optimization_success,
        'termination': _termination_text(result),
        'objective': objective,
        'candidate_objective': candidate_objective,
        'solve_time': elapsed,
        'dispatch_p_pu': dispatch_p,
        'dispatch_q_pu': dispatch_q,
        'branch_apparent_from_pu': branch_apparent_from_pu,
        'branch_apparent_to_pu': branch_apparent_to_pu,
        'branch_thermal_utilization': branch_thermal_utilization,
        'max_branch_thermal_utilization': (
            float(np.max(rated_utilization)) if rated_utilization.size else np.nan
        ),
        'branch_ge99_utilization_ratio': (
            float(np.mean(rated_utilization >= 0.99))
            if rated_utilization.size else np.nan
        ),
    }


def load_pinn(dsppc, dtheta=None):
    """Load originalcase33 FullNetand query the specifieddthetapolyhedron."""
    for path in (PRETRAIN_WEIGHTS, FULLNET_WEIGHTS):
        if not path.exists():
            raise FileNotFoundError(f'Original weight not found: {path}')

    case = TD_case.DScase_train(
        casedata=dsppc, model_type='fullnet', plot_flag=False,
        total_samples=1, batch_size=1, device=DEVICE,
    )
    dim_theta = int(case['params']['count'])
    pre_model = PreTrainNet(
        case['A_hat'], case['b_hat'], is_epigraph=False, device=DEVICE
    )
    pre_model.load_state_dict(torch.load(PRETRAIN_WEIGHTS, map_location=DEVICE))
    pre_model.eval()
    with torch.no_grad():
        A_pre, b_pre = pre_model()
    A_pre = A_pre[0].detach().cpu().numpy()
    b_pre = b_pre[0].detach().cpu().numpy()

    model = FullNet(
        dim_theta=dim_theta, A_init=A_pre, b_init=b_pre,
        is_epigraph=False, n_hidden=128, device=DEVICE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(FULLNET_WEIGHTS, map_location=DEVICE))
    model.eval()
    if dtheta is None:
        dtheta = np.zeros(dim_theta, dtype=np.float32)
    else:
        dtheta = np.asarray(dtheta, dtype=np.float32).reshape(-1)
        if dtheta.size != dim_theta:
            raise ValueError(f'dthetaThe length is{dtheta.size}, ButPINNneed{dim_theta}components')
    with torch.no_grad():
        A, b = model(torch.from_numpy(dtheta).unsqueeze(0).to(DEVICE))

    return {
        'A': A[0].detach().cpu().numpy(),
        'b': b[0].detach().cpu().numpy(),
        'dtheta': dtheta.astype(float),
        'dim_theta': dim_theta,
        'params_dict': case['params']['params_dict'],
        'errorcalculator': case['errorcalculator'],
        'fullnet_model': model,
    }


def load_nominal_pinn(dsppc):
    """Backward compatibility: Query nominal load case polyhedron."""
    return load_pinn(dsppc, dtheta=None)


def predict_pinn(pinn_template, dtheta):
    """Reuse loadedFullNetQuery newdtheta, Avoid repeated reading of weights."""
    dtheta = np.asarray(dtheta, dtype=np.float32).reshape(-1)
    if dtheta.size != pinn_template['dim_theta']:
        raise ValueError(
            f'dthetaThe length is{dtheta.size}, ButPINNneed'
            f'{pinn_template["dim_theta"]}components'
        )
    model = pinn_template['fullnet_model']
    with torch.no_grad():
        tensor = torch.from_numpy(dtheta).unsqueeze(0).to(DEVICE)
        A, b = model(tensor)
    result = dict(pinn_template)
    result.update({
        'A': A[0].detach().cpu().numpy(),
        'b': b[0].detach().cpu().numpy(),
        'dtheta': dtheta.astype(float),
    })
    return result


def uniform_dtheta(dim_theta, level):
    """allPd/QdThe channels use the same normalized perturbation."""
    if not -0.5 <= float(level) <= 0.5:
        raise ValueError('dtheta levelMust be within training range[-0.5, 0.5]')
    return np.full(dim_theta, float(level), dtype=float)


def dso_load_factor_from_level(level):
    """Original normalized inverse transformation: unificationdthetaCorresponding load magnification."""
    return 1.0 + 0.4 * float(level)


def generate_paper_dtheta_samples(dim_theta):
    """Recurrenceapproximate_polygon_coverage.pyof50Randomlydthetasequence."""
    rng = np.random.RandomState(PAPER_DTHETA_SEED)
    return rng.uniform(
        PAPER_DTHETA_RANGE[0], PAPER_DTHETA_RANGE[1],
        size=(PAPER_N_DTHETA, dim_theta),
    )


def select_coverage_percentile_conditions(coverage_rows):
    """Select by sorting position of coverage experience quantile5non-repeating operating conditions."""
    if len(coverage_rows) != PAPER_N_DTHETA:
        raise ValueError(
            f'Requirements for quantile operating condition selection{PAPER_N_DTHETA}coverage results, '
            f'Actually{len(coverage_rows)}a'
        )
    order = sorted(
        range(len(coverage_rows)),
        key=lambda index: coverage_rows[index]['coverage_percent'],
    )
    selected = []
    for percentile in GRID_COVERAGE_PERCENTILES:
        # Experience quantile ranking position: round-half-up, 50Samples do not repeat each other.
        rank = int(np.floor((percentile / 100.0) * (len(order) - 1) + 0.5))
        row_index = order[rank]
        selected.append({
            **coverage_rows[row_index],
            'coverage_percentile': int(percentile),
            'coverage_rank_zero_based': int(rank),
        })
    return selected


def dso_condition_statistics(dsppc_nominal, dsppc, dtheta):
    """Use interpretable statistics to describe full-dimensional random eventsdthetaWorking conditions."""
    p_nom = float(np.sum(dsppc_nominal['bus'][:, 2]))
    q_nom = float(np.sum(dsppc_nominal['bus'][:, 3]))
    dtheta = np.asarray(dtheta, dtype=float)
    return {
        'dso_active_load_factor': (
            float(np.sum(dsppc['bus'][:, 2])) / p_nom if abs(p_nom) > 1e-12 else np.nan
        ),
        'dso_reactive_load_factor': (
            float(np.sum(dsppc['bus'][:, 3])) / q_nom if abs(q_nom) > 1e-12 else np.nan
        ),
        'dtheta_mean': float(np.mean(dtheta)),
        'dtheta_std': float(np.std(dtheta)),
        'dtheta_min': float(np.min(dtheta)),
        'dtheta_max': float(np.max(dtheta)),
    }


def apply_dtheta_to_dsppc(dsppc_nominal, pinn_data):
    """willPINNofdthetaWrite physics according to the original denormalized caliberDSLoad."""
    result = copy.deepcopy(dsppc_nominal)
    params = pinn_data['params_dict']
    pd_init = np.asarray(params['Pd_meta']['initial_value'], dtype=float)
    qd_init = np.asarray(params['Qd_meta']['initial_value'], dtype=float)
    dtheta = np.asarray(pinn_data['dtheta'], dtype=float).reshape(-1)
    n_pd = pd_init.size
    if dtheta.size != n_pd + qd_init.size:
        raise ValueError('dthetawithPd_meta/Qd_metaDimensions do not match')
    pd_meta = pd_init + dtheta[:n_pd].reshape(pd_init.shape)
    qd_meta = qd_init + dtheta[n_pd:].reshape(qd_init.shape)
    # ReferencePointCalculatorandDScase_trainBoth use the same denormalization formula.
    result['bus'][:, 2] = (0.8 + 0.4 * pd_meta.reshape(-1)) * dsppc_nominal['bus'][:, 2]
    result['bus'][:, 3] = (0.8 + 0.4 * qd_meta.reshape(-1)) * dsppc_nominal['bus'][:, 3]
    return result


def nominal_reference_point(dsppc, pinn_data):
    calculator = ReferencePointCalculator(
        ppc=dsppc,
        init_params_dict=pinn_data['params_dict'],
        original_model=None,
        solver=SOLVER_NAME,
    )
    x_ref = calculator.compute(pinn_data['dtheta'])
    if x_ref is None:
        raise RuntimeError('Inflexible reference point power flow does not converge')
    return np.asarray(x_ref, dtype=float)


def diagnostic_current_limits(dsppc):
    """Use the same DS case to obtain the no-flexibility low-current solution and a diagnostic-only I² upper bound.

    case33branch roadRATE_A/B/CAll are0, Engineering ratings cannot be obtained directly. So fix all non-root nodes first
    The injection is the nominal load and the root node voltage is1.0, and minimize the totalI²Select the normal low current flow branch; then
    A specified multiplier of the reference current is allowed for each line, with terminal light-loaded lines allowing at least0.05 p.u.current.
    """
    model = pyo.ConcreteModel(name='diagnostic_noflex_ds')
    TD_case.DScase(model, copy.deepcopy(dsppc))
    model.diagnostic_noflex = pyo.ConstraintList()
    node_flex_dict = dsppc.get('node_flex_dict', {})
    for bus in model.BUS:
        if int(bus) == 1:
            continue
        flex_info = node_flex_dict.get(int(bus), {'type': 0})
        if not flex_info or not flex_info.get('type', 0):
            # The normal load node has been replaced byDScaseFixed to avoid adding duplicate equations resulting in too little freedom.
            continue
        model.diagnostic_noflex.add(model.Pn[bus] == -model.Pd[bus])
        model.diagnostic_noflex.add(model.Qn[bus] == -model.Qd[bus])
    model.diagnostic_noflex.add(model.V2[1] == PCC_VOLTAGE ** 2)
    model.obj = pyo.Objective(
        expr=sum(model.I2[line] for line in model.LINE), sense=pyo.minimize
    )
    solver = pyo.SolverFactory(SOLVER_NAME)
    result = solver.solve(model, tee=False)
    if not _is_success(result):
        raise RuntimeError(
            f'No flexibility in generating upper diagnostic current boundsDSNot convergent: {_termination_text(result)}'
        )
    i2_reference = np.array(
        [pyo.value(model.I2[line]) for line in model.LINE], dtype=float
    )
    current_reference = np.sqrt(np.maximum(i2_reference, 0.0))
    current_limit = np.maximum(
        DIAGNOSTIC_CURRENT_MULTIPLIER * current_reference,
        DIAGNOSTIC_CURRENT_FLOOR_PU,
    )
    return i2_reference, current_limit ** 2


def make_host_ts_case(tsppc_original, dsppc, x_ref):
    """From the originalTSNominal peeling under loadPCCpower; add backx_refStrictly restore the original load when."""
    host = copy.deepcopy(tsppc_original)
    scale = float(dsppc['baseMVA'])
    host['bus'][PCC_NODE, 2] -= x_ref[0] * scale
    host['bus'][PCC_NODE, 3] -= x_ref[1] * scale
    return host


def scale_host_ts_load(host_tsppc, load_factor):
    """Only scale native host-TS loads; keep the DSO nominal condition fixed."""
    scaled = copy.deepcopy(host_tsppc)
    scaled['bus'][:, 2] *= float(load_factor)
    scaled['bus'][:, 3] *= float(load_factor)
    return scaled


def compute_nominal_coverage(dsppc, pinn_data, n_dirs=COVERAGE_N_DIRS,
                             direction_seed=0):
    """Reuseapproximate_polygon_coverage.pyThe radial ratio diameter of."""
    directions = generate_random_directions(n_dirs, seed=direction_seed)
    ec = pinn_data['errorcalculator']
    x_ref = nominal_reference_point(dsppc, pinn_data)
    A = np.asarray(pinn_data['A'], dtype=float)
    b = np.asarray(pinn_data['b'], dtype=float)

    member, violation = reference_point_membership(
        A, b, x_ref, REFERENCE_MEMBERSHIP_TOL
    )
    if not member:
        raise RuntimeError(
            f'Nominal reference point is not presentPINNwithin polyhedron: max violation={violation:.3e}'
        )

    ray_model = create_ray_model(ec)
    update_ray_model_params(
        ray_model, pinn_data['params_dict'], pinn_data['dtheta']
    )
    solver = pyo.SolverFactory(SOLVER_NAME)
    k_true, k_poly, valid_directions = [], [], []
    for direction in directions:
        true_radius = compute_k_max_ray(ray_model, x_ref, direction, solver)
        if not np.isfinite(true_radius) or true_radius <= 1e-12:
            continue
        poly_radius = compute_k_max_polytope(A, b, x_ref, direction)
        if not np.isfinite(poly_radius):
            continue
        k_true.append(float(true_radius))
        k_poly.append(float(poly_radius))
        valid_directions.append(direction)

    k_true = np.asarray(k_true, dtype=float)
    k_poly = np.asarray(k_poly, dtype=float)
    valid_directions = np.asarray(valid_directions, dtype=float)
    if k_true.size != n_dirs:
        raise RuntimeError(
            f'Coverage only gets{k_true.size}/{n_dirs}valid direction'
        )
    ratios = k_poly / k_true

    convergence = []
    checkpoints = [value for value in (100, 360, 720) if value <= n_dirs]
    if n_dirs not in checkpoints:
        checkpoints.append(n_dirs)
    for count in checkpoints:
        metrics = summarize_coverage_ratios(ratios[:count])
        convergence.append({'n_directions': count, **metrics})
    return convergence[-1], {
        'directions': valid_directions,
        'x_ref': x_ref,
        'k_true': k_true,
        'k_pinn': k_poly,
        'coverage_ratio': ratios,
    }, convergence


def fix_pcc_voltage(model):
    model.V[PCC_NODE].fix(PCC_VOLTAGE)


def build_base_model(host_tsppc, dsppc, x_ref):
    # inPCCNode releasePd/Qd, Use the same againhostLoad+x_refFixed, keeping the same caliber as the other two models.
    model = TD_case.TScase(
        tscasedata=copy.deepcopy(host_tsppc),
        dscasedata_dict={PCC_NODE: {}},
    )
    ts_base = float(host_tsppc['baseMVA'])
    ds_base = float(dsppc['baseMVA'])
    pd_host = float(host_tsppc['bus'][PCC_NODE, 2] / ts_base)
    qd_host = float(host_tsppc['bus'][PCC_NODE, 3] / ts_base)
    model.base_pcc_constraints = pyo.ConstraintList()
    model.base_pcc_constraints.add(
        model.Pd[PCC_NODE] == pd_host + x_ref[0] * ds_base / ts_base
    )
    model.base_pcc_constraints.add(
        model.Qd[PCC_NODE] == qd_host + x_ref[1] * ds_base / ts_base
    )
    fix_pcc_voltage(model)
    return model


def build_pinn_model(host_tsppc, dsppc, pinn_data):
    apx_data = {
        'baseMVA': dsppc['baseMVA'],
        'A_hat': pinn_data['A'],
        'b_hat': pinn_data['b'],
    }
    model = TD_case.TDcase(
        tscasedata=copy.deepcopy(host_tsppc),
        dscasedata_dict={PCC_NODE: apx_data},
        is_apx=True,
    )
    # Under this protocol, the PCC voltage is fixed at 1.0; DScase_apx does not use V2 for region parameterization.
    fix_pcc_voltage(model)
    return model


def build_true_model(host_tsppc, dsppc, i2_limits=None):
    model = TD_case.TDcase(
        tscasedata=copy.deepcopy(host_tsppc),
        dscasedata_dict={PCC_NODE: copy.deepcopy(dsppc)},
        is_apx=False,
    )
    fix_pcc_voltage(model)
    if i2_limits is not None:
        block = model.DS[PCC_NODE]
        block.diagnostic_current_limits = pyo.ConstraintList()
        for line, i2_max in zip(block.LINE, i2_limits):
            block.diagnostic_current_limits.add(block.I2[line] <= float(i2_max))
    return model


def initialize_true_model(model, start_index):
    """Apply five deterministic, deliberately different NLP initializations."""
    rng = np.random.RandomState(1000 + start_index)
    voltage_start = (0.96, 0.99, 1.00, 1.02, 1.04)[start_index]
    flow_scale = (0.15, 0.30, 0.45, 0.60, 0.80)[start_index]
    for var in model.component_data_objects(pyo.Var, descend_into=True):
        if var.fixed:
            continue
        name = var.name
        if '.V2[' in name:
            value = voltage_start ** 2
        elif name.startswith('V['):
            value = voltage_start
        elif 'theta[' in name:
            value = float(rng.uniform(-0.08, 0.08))
        elif '.I2[' in name:
            value = max(flow_scale ** 2, 1e-4)
        elif '.Pf[' in name:
            value = flow_scale * float(rng.uniform(0.7, 1.3))
        elif '.Qf[' in name:
            value = 0.6 * flow_scale * float(rng.uniform(0.7, 1.3))
        elif '.Pn[' in name or '.Qn[' in name:
            value = 0.0
        elif name.startswith('Pg[') or name.startswith('Qg['):
            if var.lb is not None and var.ub is not None:
                value = float(var.lb + (0.2 + 0.15 * start_index)
                              * (var.ub - var.lb))
            else:
                value = 0.0
        elif var.value is not None:
            value = float(var.value)
        else:
            value = 0.0
        if var.lb is not None:
            value = max(value, float(var.lb) + 1e-8)
        if var.ub is not None:
            value = min(value, float(var.ub) - 1e-8)
        var.set_value(value)


def run_true_multistart():
    tsppc = getattr(TD_case, TS_CASE)()
    dsppc = getattr(TD_case, DS_CASE)(root_voltage=PCC_VOLTAGE)
    pinn_data = load_nominal_pinn(dsppc)
    x_ref = nominal_reference_point(dsppc, pinn_data)
    host = make_host_ts_case(tsppc, dsppc, x_ref)
    rows = []
    for start_index in range(TRUE_MULTISTART_COUNT):
        model = build_true_model(host, dsppc, i2_limits=None)
        initialize_true_model(model, start_index)
        solved = solve_model(model)
        pcc = (interface_point(model, 'true', x_ref)
               if solved['success'] else np.full(2, np.nan))
        rows.append({
            'start_index': start_index,
            'success': bool(solved['success']),
            'feasible_solution': bool(solved['feasible_solution']),
            'optimization_success': bool(solved['optimization_success']),
            'termination': solved['termination'],
            'cost': solved['objective'],
            'candidate_cost': solved['candidate_objective'],
            'pcc_p_pu_dsbase': pcc[0],
            'pcc_q_pu_dsbase': pcc[1],
            'solve_time_s': solved['solve_time'],
        })
        print(f"Multiple initial values {start_index + 1}/{TRUE_MULTISTART_COUNT}: "
              f"success={solved['success']} cost={solved['objective']:.9f} "
              f"PCC={pcc}")
    return rows


def run_representative_multistart(scenario_name, host_tsppc, dsppc,
                                  pinn_data, x_ref, metadata):
    """Five starts for Base/PINN/True under one representative scenario."""
    builders = {
        'base': lambda: build_base_model(host_tsppc, dsppc, x_ref),
        'pinn': lambda: build_pinn_model(host_tsppc, dsppc, pinn_data),
        'true': lambda: build_true_model(host_tsppc, dsppc, i2_limits=None),
    }
    rows = []
    for region, builder in builders.items():
        for start_index in range(TRUE_MULTISTART_COUNT):
            model = builder()
            initialize_true_model(model, start_index)
            solved = solve_model(model)
            pcc = (
                interface_point(model, region, x_ref)
                if solved['feasible_solution'] else np.full(2, np.nan)
            )
            row = {
                'scenario_name': scenario_name,
                'region': region,
                'start_index': start_index,
                'feasible_solution': bool(solved['feasible_solution']),
                'optimization_success': bool(solved['optimization_success']),
                'termination': solved['termination'],
                'cost': solved['objective'],
                'candidate_cost': solved['candidate_objective'],
                'pcc_p_pu_dsbase': pcc[0],
                'pcc_q_pu_dsbase': pcc[1],
                'solve_time_s': solved['solve_time'],
                **metadata,
            }
            rows.append(row)
            print(
                f"Multiple initial values {scenario_name}/{region} "
                f"{start_index + 1}/{TRUE_MULTISTART_COUNT}: "
                f"feasible={row['feasible_solution']} "
                f"optimal={row['optimization_success']} "
                f"cost={row['cost']:.9f} PCC={pcc}"
            )
    return rows


def interface_point(model, kind, x_ref):
    if kind == 'base':
        return np.asarray(x_ref, dtype=float)
    block = model.DS[PCC_NODE]
    return np.array([pyo.value(block.Pn[1]), pyo.value(block.Qn[1])], dtype=float)


def disaggregation_check(x_pinn, dsppc):
    """back generationPINNScheduling points and explicitly returning solution status and squared projection error."""
    try:
        model = pyo.ConcreteModel(name='pinn_ds_disaggregation')
        TD_case.DScase(model, copy.deepcopy(dsppc))
        model.constraints.add(model.V2[1] == PCC_VOLTAGE ** 2)
        model.obj = pyo.Objective(
            expr=(float(x_pinn[0]) - model.Pn[1]) ** 2
            + (float(x_pinn[1]) - model.Qn[1]) ** 2,
            sense=pyo.minimize,
        )
        solver = pyo.SolverFactory(SOLVER_NAME)
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            result = solver.solve(model, tee=False)
        elapsed = time.perf_counter() - start
        feasible = _has_feasible_solution(result)
        optimal = _is_optimization_success(result)
        return {
            'squared_error': (
                float(pyo.value(model.obj)) if optimal else np.nan
            ),
            'feasible_solution': bool(feasible),
            'optimization_success': bool(optimal),
            'termination': _termination_text(result),
            'solve_time_s': elapsed,
        }
    except Exception as exc:
        print(f'[warning] PINNScheduling point back to realDSfailed: {exc}')
        return {
            'squared_error': np.nan,
            'feasible_solution': False,
            'optimization_success': False,
            'termination': f'exception: {type(exc).__name__}: {exc}',
            'solve_time_s': np.nan,
        }


def save_results(rows, arrays, diagnostics):
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        'region', 'success', 'feasible_solution', 'optimization_success',
        'termination', 'cost', 'candidate_cost', 'solve_time_s',
        'pcc_p_pu_dsbase', 'pcc_q_pu_dsbase',
        'max_tso_branch_utilization_percent',
        'tso_branch_ge99_utilization_ratio_percent',
        'pinn_disaggregation_squared_error', 'pinn_disaggregation_feasible',
        'pinn_disaggregation_optimization_success',
        'pinn_disaggregation_termination',
    ]
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        OUT / 'raw_results.npz',
        regions=np.asarray([r['region'] for r in rows]),
        costs=np.asarray([r['cost'] for r in rows], dtype=float),
        candidate_costs=np.asarray([r['candidate_cost'] for r in rows], dtype=float),
        success=np.asarray([r['success'] for r in rows], dtype=bool),
        feasible_solution=np.asarray(
            [r['feasible_solution'] for r in rows], dtype=bool),
        optimization_success=np.asarray(
            [r['optimization_success'] for r in rows], dtype=bool),
        solve_time_s=np.asarray([r['solve_time_s'] for r in rows], dtype=float),
        pcc=np.stack([arrays[r['region']]['pcc'] for r in rows]),
        dispatch_p=np.stack([arrays[r['region']]['dispatch_p'] for r in rows]),
        dispatch_q=np.stack([arrays[r['region']]['dispatch_q'] for r in rows]),
        tso_branch_apparent_from_pu=np.stack([
            arrays[r['region']]['branch_apparent_from_pu'] for r in rows]),
        tso_branch_apparent_to_pu=np.stack([
            arrays[r['region']]['branch_apparent_to_pu'] for r in rows]),
        tso_branch_thermal_utilization=np.stack([
            arrays[r['region']]['branch_thermal_utilization'] for r in rows]),
        A_pinn=diagnostics['A_pinn'], b_pinn=diagnostics['b_pinn'],
        x_ref=diagnostics['x_ref'],
        reference_member=np.asarray(diagnostics['reference_member']),
        reference_max_violation=np.asarray(diagnostics['reference_max_violation']),
        diagnostic_i2_reference=diagnostics['diagnostic_i2_reference'],
        diagnostic_i2_limits=diagnostics['diagnostic_i2_limits'],
        true_i2=diagnostics['true_i2'],
        true_current_limit_active_rate=np.asarray(
            diagnostics['true_current_limit_active_rate']),
        diagnostic_current_multiplier=np.asarray(DIAGNOSTIC_CURRENT_MULTIPLIER),
        diagnostic_current_floor_pu=np.asarray(DIAGNOSTIC_CURRENT_FLOOR_PU),
        ts_load_factor=np.asarray(TS_LOAD_FACTOR),
        dso_dtheta_level=np.asarray(diagnostics['dso_dtheta_level']),
        dso_scenario_index=np.asarray(diagnostics['dso_scenario_index']),
        dso_active_load_factor=np.asarray(diagnostics['dso_active_load_factor']),
        dso_reactive_load_factor=np.asarray(diagnostics['dso_reactive_load_factor']),
        dtheta=np.asarray(diagnostics['dtheta'], dtype=float),
        value_retention_percent=np.asarray(diagnostics['value_retention_percent']),
        cost_order_ok=np.asarray(diagnostics['cost_order_ok']),
        pcc_voltage=np.asarray(PCC_VOLTAGE),
        dim_theta=np.asarray(diagnostics['dim_theta']),
    )


def run_single_scenario(dso_dtheta_level=0.0, pinn_data=None,
                        dsppc_nominal=None, dso_scenario_index=-1,
                        nominal_host_tsppc=None):
    print(f'Equipment: {DEVICE}  TSO={TS_CASE}  DSO={DS_CASE}  PCCNode index={PCC_NODE}')
    print(f'PCCVoltage fixed: {PCC_VOLTAGE:.3f} p.u.')
    tsppc_original = getattr(TD_case, TS_CASE)()
    if dsppc_nominal is None:
        dsppc_nominal = getattr(TD_case, DS_CASE)(root_voltage=PCC_VOLTAGE)
    if pinn_data is None:
        probe = load_nominal_pinn(dsppc_nominal)
        dtheta = uniform_dtheta(probe['dim_theta'], dso_dtheta_level)
        pinn_data = load_pinn(dsppc_nominal, dtheta=dtheta)
    dsppc = apply_dtheta_to_dsppc(dsppc_nominal, pinn_data)
    condition = dso_condition_statistics(
        dsppc_nominal, dsppc, pinn_data['dtheta']
    )
    print(f'TSO hostLoad rate: {TS_LOAD_FACTOR:.2f}; '
          f'DSOsample: {dso_scenario_index + 1 if dso_scenario_index >= 0 else "uniform"}; '
          f'P/QTotal load ratio: {condition["dso_active_load_factor"]:.4f}/'
          f'{condition["dso_reactive_load_factor"]:.4f}')
    # The reference-point calculator applies dtheta internally, so pass the unperturbed nominal case.
    x_ref = nominal_reference_point(dsppc_nominal, pinn_data)
    is_member, max_violation = reference_point_membership(
        pinn_data['A'], pinn_data['b'], x_ref, REFERENCE_MEMBERSHIP_TOL
    )
    print(f'PINNinput dimensions: {pinn_data["dim_theta"]}; '
          f'dtheta mean/std={condition["dtheta_mean"]:+.4f}/'
          f'{condition["dtheta_std"]:.4f}')
    print(f'No flexibility reference point: [P,Q]={x_ref}')
    print(f'The reference point belongs toPINNPolyhedron: {is_member}, maximum violation={max_violation:.3e}')
    if USE_DSO_CURRENT_LIMITS:
        i2_reference, i2_limits = diagnostic_current_limits(dsppc)
        print('Diagnostic current limit: Each line does not exceed the inflexibility reference current'
              f'{DIAGNOSTIC_CURRENT_MULTIPLIER:.1f}times, '
              f'minimum limit={DIAGNOSTIC_CURRENT_FLOOR_PU:.3f} p.u.')
        print(f'Reference maximum current={np.sqrt(i2_reference.max()):.4f} p.u., '
              f'Maximum allowable current={np.sqrt(i2_limits.max()):.4f} p.u.')
    else:
        i2_reference = np.full(len(dsppc['branch']), np.nan)
        i2_limits = None
        print('DSOLine Capacity Constraint: Off (with originalPINNTrain physical models to be consistent) ')

    if nominal_host_tsppc is None:
        # The host/DSO decomposition is defined once at the nominal DSO
        # condition. It must not be recomputed from the current DSO sample,
        # otherwise a larger DSO load is cancelled by a smaller host load.
        nominal_pinn = load_nominal_pinn(dsppc_nominal)
        nominal_x_ref = nominal_reference_point(dsppc_nominal, nominal_pinn)
        nominal_host_tsppc = make_host_ts_case(
            tsppc_original, dsppc_nominal, nominal_x_ref)
    nominal_host_pcc_p_mw = float(nominal_host_tsppc['bus'][PCC_NODE, 2])
    nominal_host_pcc_q_mvar = float(nominal_host_tsppc['bus'][PCC_NODE, 3])
    host_tsppc = copy.deepcopy(nominal_host_tsppc)
    host_tsppc = scale_host_ts_load(host_tsppc, TS_LOAD_FACTOR)
    models = {
        'base': build_base_model(host_tsppc, dsppc, x_ref),
        'pinn': build_pinn_model(host_tsppc, dsppc, pinn_data),
        'true': build_true_model(host_tsppc, dsppc, i2_limits=i2_limits),
    }

    rows, arrays = [], {}
    for region in ('base', 'pinn', 'true'):
        print(f'\n[{region}] Start solving')
        solved = solve_model(models[region])
        pcc = (interface_point(models[region], region, x_ref)
               if solved['success'] else np.full(2, np.nan))
        arrays[region] = {
            'pcc': pcc,
            'dispatch_p': solved['dispatch_p_pu'],
            'dispatch_q': solved['dispatch_q_pu'],
            'branch_apparent_from_pu': solved['branch_apparent_from_pu'],
            'branch_apparent_to_pu': solved['branch_apparent_to_pu'],
            'branch_thermal_utilization': solved['branch_thermal_utilization'],
        }
        row = {
            'region': region,
            'success': bool(solved['success']),
            'feasible_solution': bool(solved['feasible_solution']),
            'optimization_success': bool(solved['optimization_success']),
            'termination': solved['termination'],
            'cost': solved['objective'],
            'candidate_cost': solved['candidate_objective'],
            'solve_time_s': solved['solve_time'],
            'pcc_p_pu_dsbase': pcc[0],
            'pcc_q_pu_dsbase': pcc[1],
            'max_tso_branch_utilization_percent': (
                100.0 * solved['max_branch_thermal_utilization']
            ),
            'tso_branch_ge99_utilization_ratio_percent': (
                100.0 * solved['branch_ge99_utilization_ratio']
            ),
            'pinn_disaggregation_squared_error': np.nan,
            'pinn_disaggregation_feasible': False,
            'pinn_disaggregation_optimization_success': False,
            'pinn_disaggregation_termination': '',
        }
        rows.append(row)
        print(f"  success={row['success']} termination={row['termination']}")
        print(f"  cost={row['cost']:.8g} time={row['solve_time_s']:.3f}s PCC={pcc}")
        print('  TSOMaximum line utilization='
              f'{row["max_tso_branch_utilization_percent"]:.3f}%, '
              'Utilization>=99%rated line ratio='
              f'{row["tso_branch_ge99_utilization_ratio_percent"]:.1f}%')

    disagg = (
        disaggregation_check(arrays['pinn']['pcc'], dsppc)
        if rows[1]['optimization_success']
        else {
            'squared_error': np.nan,
            'feasible_solution': False,
            'optimization_success': False,
            'termination': 'PINN economic optimization not successful',
            'solve_time_s': np.nan,
        }
    )
    pinn_disagg = disagg['squared_error']
    rows[1]['pinn_disaggregation_squared_error'] = pinn_disagg
    rows[1]['pinn_disaggregation_feasible'] = disagg['feasible_solution']
    rows[1]['pinn_disaggregation_optimization_success'] = \
        disagg['optimization_success']
    rows[1]['pinn_disaggregation_termination'] = disagg['termination']

    true_i2 = (
        np.array([pyo.value(models['true'].DS[PCC_NODE].I2[line])
                  for line in models['true'].DS[PCC_NODE].LINE], dtype=float)
        if rows[2]['success'] else np.full(len(dsppc['branch']), np.nan)
    )
    true_active_rate = (
        float(np.mean(true_i2 >= 0.99 * i2_limits))
        if i2_limits is not None and np.all(np.isfinite(true_i2)) else np.nan
    )

    costs = {row['region']: float(row['cost']) for row in rows}
    all_success = all(row['optimization_success'] for row in rows)
    cost_order_ok = bool(
        all_success
        and costs['true'] <= costs['pinn'] + 1e-6
        and costs['pinn'] <= costs['base'] + 1e-6
    )
    denominator = costs['base'] - costs['true']
    value_retention = (
        100.0 * (costs['base'] - costs['pinn']) / denominator
        if all_success and abs(denominator) > 1e-12 else np.nan
    )
    diagnostics = {
        'A_pinn': pinn_data['A'], 'b_pinn': pinn_data['b'],
        'x_ref': x_ref, 'reference_member': is_member,
        'reference_max_violation': max_violation,
        'diagnostic_i2_reference': i2_reference,
        'diagnostic_i2_limits': (
            i2_limits if i2_limits is not None
            else np.full(len(dsppc['branch']), np.nan)
        ),
        'true_i2': true_i2,
        'true_current_limit_active_rate': true_active_rate,
        'value_retention_percent': value_retention,
        'cost_order_ok': cost_order_ok,
        'dim_theta': pinn_data['dim_theta'],
        'dso_dtheta_level': float(dso_dtheta_level),
        'dso_scenario_index': int(dso_scenario_index),
        **condition,
        'dtheta': pinn_data['dtheta'],
    }
    save_results(rows, arrays, diagnostics)

    print('\n=== Diagnosis summary ===')
    print(f"cost: true={costs['true']:.8g}, pinn={costs['pinn']:.8g}, base={costs['base']:.8g}")
    print(f'satisfy C_true <= C_PINN <= C_base: {cost_order_ok}')
    print(f'economic value retention rate: {value_retention:.3f}%')
    print(f'PINNScheduling point is realDSsquared projection error: {pinn_disagg:.3e}')
    print('PINNSubstitute to solve: '
          f'optimal={disagg["optimization_success"]}, '
          f'termination={disagg["termination"]}')
    print(f'trueDSProportion of lines touching the upper diagnostic current limit: {true_active_rate:.1%}')
    print(f'result: {OUT / "summary.csv"}')
    print(f'raw data: {OUT / "raw_results.npz"}')

    return {
        'ts_load_factor': TS_LOAD_FACTOR,
        'dso_dtheta_level': float(dso_dtheta_level),
        'dso_scenario_index': int(dso_scenario_index),
        # This alias is reserved for old plotting code to read and its meaning is the total active load multiplier.
        'dso_load_factor': condition['dso_active_load_factor'],
        **condition,
        'dso_current_limits_enabled': USE_DSO_CURRENT_LIMITS,
        'nominal_host_pcc_p_mw': nominal_host_pcc_p_mw,
        'nominal_host_pcc_q_mvar': nominal_host_pcc_q_mvar,
        'current_multiplier': DIAGNOSTIC_CURRENT_MULTIPLIER,
        'base_cost': costs['base'],
        'pinn_cost': costs['pinn'],
        'true_cost': costs['true'],
        'base_solve_time_s': float(rows[0]['solve_time_s']),
        'pinn_solve_time_s': float(rows[1]['solve_time_s']),
        'true_solve_time_s': float(rows[2]['solve_time_s']),
        'pinn_true_cost_gap': costs['pinn'] - costs['true'],
        'pinn_true_relative_cost_gap_percent': (
            100.0 * (costs['pinn'] - costs['true']) / costs['true']
            if all_success and abs(costs['true']) > 1e-12 else np.nan
        ),
        'true_flexibility_value': costs['base'] - costs['true'],
        'pinn_flexibility_value': costs['base'] - costs['pinn'],
        'value_retention_percent': value_retention,
        'cost_order_ok': cost_order_ok,
        'all_economic_optimization_success': all_success,
        'base_feasible_solution': bool(rows[0]['feasible_solution']),
        'pinn_feasible_solution': bool(rows[1]['feasible_solution']),
        'true_feasible_solution': bool(rows[2]['feasible_solution']),
        'base_optimization_success': bool(rows[0]['optimization_success']),
        'pinn_optimization_success': bool(rows[1]['optimization_success']),
        'true_optimization_success': bool(rows[2]['optimization_success']),
        'pinn_disaggregation_squared_error': pinn_disagg,
        'pinn_disaggregation_solved': bool(disagg['optimization_success']),
        'pinn_disaggregation_feasible': bool(disagg['feasible_solution']),
        'pinn_disaggregation_optimization_success': bool(
            disagg['optimization_success']),
        'pinn_disaggregation_termination': disagg['termination'],
        'pinn_disaggregation_solve_time_s': disagg['solve_time_s'],
        'pinn_true_pcc_distance_pu': float(
            np.linalg.norm(arrays['pinn']['pcc'] - arrays['true']['pcc'])
        ),
        'pinn_true_pg_l1_mw': float(
            np.sum(np.abs(arrays['pinn']['dispatch_p']
                          - arrays['true']['dispatch_p']))
            * tsppc_original['baseMVA']
        ),
        'pinn_true_pg_linf_mw': float(
            np.max(np.abs(arrays['pinn']['dispatch_p']
                          - arrays['true']['dispatch_p']))
            * tsppc_original['baseMVA']
        ),
        'pinn_true_qg_l1_mvar': float(
            np.sum(np.abs(arrays['pinn']['dispatch_q']
                          - arrays['true']['dispatch_q']))
            * tsppc_original['baseMVA']
        ),
        'pinn_true_qg_linf_mvar': float(
            np.max(np.abs(arrays['pinn']['dispatch_q']
                          - arrays['true']['dispatch_q']))
            * tsppc_original['baseMVA']
        ),
        'true_current_limit_active_rate': true_active_rate,
        'reference_max_current_pu': (
            float(np.sqrt(np.nanmax(i2_reference)))
            if USE_DSO_CURRENT_LIMITS else np.nan
        ),
        'allowed_max_current_pu': (
            float(np.sqrt(np.max(i2_limits))) if i2_limits is not None else np.nan
        ),
        'true_pcc_p_pu_dsbase': float(arrays['true']['pcc'][0]),
        'true_pcc_q_pu_dsbase': float(arrays['true']['pcc'][1]),
        'base_max_tso_branch_utilization_percent': float(
            rows[0]['max_tso_branch_utilization_percent']),
        'pinn_max_tso_branch_utilization_percent': float(
            rows[1]['max_tso_branch_utilization_percent']),
        'true_max_tso_branch_utilization_percent': float(
            rows[2]['max_tso_branch_utilization_percent']),
        'base_tso_branch_ge99_utilization_ratio_percent': float(
            rows[0]['tso_branch_ge99_utilization_ratio_percent']),
        'pinn_tso_branch_ge99_utilization_ratio_percent': float(
            rows[1]['tso_branch_ge99_utilization_ratio_percent']),
        'true_tso_branch_ge99_utilization_ratio_percent': float(
            rows[2]['tso_branch_ge99_utilization_ratio_percent']),
    }


def save_sensitivity_summary(rows):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / 'current_limit_sensitivity.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        OUT_ROOT / 'current_limit_sensitivity.npz',
        **{key: np.asarray([row[key] for row in rows]) for key in rows[0]},
    )
    return path


def run_current_sensitivity():
    global DIAGNOSTIC_CURRENT_MULTIPLIER, TS_LOAD_FACTOR, USE_DSO_CURRENT_LIMITS, OUT
    TS_LOAD_FACTOR = 1.0
    USE_DSO_CURRENT_LIMITS = True
    sensitivity_rows = []
    for multiplier in DIAGNOSTIC_CURRENT_MULTIPLIERS:
        DIAGNOSTIC_CURRENT_MULTIPLIER = float(multiplier)
        OUT = OUT_ROOT / f'current_multiplier_{multiplier:.1f}'
        print('\n' + '=' * 72)
        print(f'Current upper limit rate sensitivity check: kappa={multiplier:.1f}')
        print('=' * 72)
        sensitivity_rows.append(run_single_scenario())

    OUT = OUT_ROOT
    summary_path = save_sensitivity_summary(sensitivity_rows)
    print('\n=== Summary of current upper limit rate sensitivity ===')
    for row in sensitivity_rows:
        print(
            f"kappa={row['current_multiplier']:.1f} "
            f"true_cost={row['true_cost']:.6f} "
            f"PINN-true={row['pinn_true_cost_gap']:.6f} "
            f"retention={row['value_retention_percent']:.3f}% "
            f"active={row['true_current_limit_active_rate']:.1%}"
        )
    print(f'Sensitivity summary: {summary_path}')


def precheck_load_factors():
    """Solve Base only before the complete multi-scenario experiment."""
    tsppc_original = getattr(TD_case, TS_CASE)()
    dsppc = getattr(TD_case, DS_CASE)(root_voltage=PCC_VOLTAGE)
    pinn_data = load_nominal_pinn(dsppc)
    x_ref = nominal_reference_point(dsppc, pinn_data)
    nominal_host = make_host_ts_case(tsppc_original, dsppc, x_ref)
    rows = []
    for factor in TS_LOAD_FACTORS:
        host = scale_host_ts_load(nominal_host, factor)
        solved = solve_model(build_base_model(host, dsppc, x_ref))
        rows.append({
            'ts_load_factor': factor,
            'success': bool(solved['success']),
            'termination': solved['termination'],
            'base_cost': solved['objective'],
            'solve_time_s': solved['solve_time'],
        })
        print(
            f"pre-check alpha={factor:.2f}: success={solved['success']} "
            f"termination={solved['termination']} cost={solved['objective']:.6f}"
        )
    out_dir = OUT_ROOT / 'load_scenarios'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'base_precheck.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return [row['ts_load_factor'] for row in rows if row['success']], path


def save_load_scenario_summary(rows):
    out_dir = OUT_ROOT / 'load_scenarios'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'scenario_summary.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out_dir / 'scenario_raw_results.npz',
        **{key: np.asarray([row[key] for row in rows]) for key in rows[0]},
    )
    return path


def save_operating_grid(rows, coverage_rows, selected_coverage_rows,
                        coverage_raw_by_dso, multistart_rows):
    """Baozi50Working condition coverage and5x5Downstream dispatch evidence."""
    out_dir = OUT_ROOT / GRID_OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'grid_summary.csv'
    with summary_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out_dir / 'grid_raw_results.npz',
        **{key: np.asarray([row[key] for row in rows]) for key in rows[0]},
    )

    coverage_path = out_dir / 'coverage_all_50_dso.csv'
    with coverage_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coverage_rows)
    selected_path = out_dir / 'selected_dso_conditions.csv'
    with selected_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_coverage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_coverage_rows)

    raw_indices = sorted(coverage_raw_by_dso)
    np.savez_compressed(
        out_dir / 'coverage_raw_all_50_dso.npz',
        dso_scenario_index=np.asarray(raw_indices, dtype=int),
        dtheta=np.stack([coverage_raw_by_dso[i]['dtheta'] for i in raw_indices]),
        directions=np.stack([
            coverage_raw_by_dso[i]['directions'] for i in raw_indices]),
        x_ref=np.stack([coverage_raw_by_dso[i]['x_ref'] for i in raw_indices]),
        k_true=np.stack([coverage_raw_by_dso[i]['k_true'] for i in raw_indices]),
        k_pinn=np.stack([coverage_raw_by_dso[i]['k_pinn'] for i in raw_indices]),
        coverage_ratio=np.stack([
            coverage_raw_by_dso[i]['coverage_ratio'] for i in raw_indices]),
        dtheta_generation_seed=np.asarray(PAPER_DTHETA_SEED),
        direction_seed=np.asarray(raw_indices, dtype=int),
        dtheta_range=np.asarray(PAPER_DTHETA_RANGE),
        definition=np.asarray(
            'same as approximate_polygon_coverage.py: '
            '50 dtheta x 100 directions; rho=k_pinn/k_true'
        ),
    )

    with (out_dir / 'representative_multistart.csv').open(
            'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(multistart_rows[0].keys()))
        writer.writeheader()
        writer.writerows(multistart_rows)

    multistart_summary = []
    group_keys = sorted({
        (row['scenario_name'], row['region']) for row in multistart_rows
    })
    for scenario_name, region in group_keys:
        group = [row for row in multistart_rows
                 if row['scenario_name'] == scenario_name
                 and row['region'] == region]
        optimal = [row for row in group if row['optimization_success']]
        costs = np.asarray([row['cost'] for row in optimal], dtype=float)
        pcc = np.asarray([
            [row['pcc_p_pu_dsbase'], row['pcc_q_pu_dsbase']]
            for row in optimal
        ], dtype=float)
        max_pcc_distance = 0.0
        for i in range(len(pcc)):
            for j in range(i + 1, len(pcc)):
                max_pcc_distance = max(
                    max_pcc_distance, float(np.linalg.norm(pcc[i] - pcc[j])))
        multistart_summary.append({
            'scenario_name': scenario_name,
            'region': region,
            'feasible_count': sum(row['feasible_solution'] for row in group),
            'optimization_success_count': len(optimal),
            'total_start_count': len(group),
            'cost_min': float(np.min(costs)) if costs.size else np.nan,
            'cost_max': float(np.max(costs)) if costs.size else np.nan,
            'cost_range': float(np.ptp(costs)) if costs.size else np.nan,
            'max_pcc_distance_pu': max_pcc_distance if costs.size else np.nan,
        })
    with (out_dir / 'representative_multistart_summary.csv').open(
            'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f, fieldnames=list(multistart_summary[0].keys()))
        writer.writeheader()
        writer.writerows(multistart_summary)

    successful = [row for row in rows
                  if np.isfinite(row['true_cost']) and np.isfinite(row['pinn_cost'])]
    all_ratios = np.concatenate([
        coverage_raw_by_dso[i]['coverage_ratio'] for i in raw_indices
    ])
    multistart_cost_ranges = [
        row['cost_range'] for row in multistart_summary
        if np.isfinite(row['cost_range'])
    ]
    aggregate = {
        'coverage_dso_condition_count': len(coverage_rows),
        'coverage_direction_count_per_condition': PAPER_N_DIRS,
        'coverage_raw_sample_count': int(all_ratios.size),
        'overall_coverage_percent': 100.0 * float(np.mean(all_ratios)),
        'overall_undercoverage_percent': 100.0 * float(
            np.mean(np.maximum(1.0 - all_ratios, 0.0))),
        'overall_overcoverage_percent': 100.0 * float(
            np.mean(np.maximum(all_ratios - 1.0, 0.0))),
        'selected_dso_condition_count': len(selected_coverage_rows),
        'scenario_count': len(rows),
        'successful_scenario_count': len(successful),
        'pinn_disaggregation_optimization_success_count': sum(
            bool(row['pinn_disaggregation_optimization_success'])
            for row in rows),
        'pinn_disaggregation_feasible_count': sum(
            bool(row['pinn_disaggregation_feasible']) for row in rows),
        'cost_order_success_count': sum(bool(row['cost_order_ok']) for row in rows),
        # Keep the 50-condition screening range distinct from the five
        # percentile conditions actually sent to the downstream optimizer.
        'all_50_coverage_min_percent': min(
            row['coverage_percent'] for row in coverage_rows),
        'all_50_coverage_max_percent': max(
            row['coverage_percent'] for row in coverage_rows),
        'all_50_undercoverage_min_percent': min(
            row['undercoverage_percent'] for row in coverage_rows),
        'all_50_undercoverage_max_percent': max(
            row['undercoverage_percent'] for row in coverage_rows),
        'all_50_overcoverage_max_percent': max(
            row['overcoverage_percent'] for row in coverage_rows),
        'selected_5_coverage_min_percent': min(
            row['coverage_percent'] for row in selected_coverage_rows),
        'selected_5_coverage_max_percent': max(
            row['coverage_percent'] for row in selected_coverage_rows),
        'selected_5_undercoverage_min_percent': min(
            row['undercoverage_percent'] for row in selected_coverage_rows),
        'selected_5_undercoverage_max_percent': max(
            row['undercoverage_percent'] for row in selected_coverage_rows),
        'selected_5_overcoverage_max_percent': max(
            row['overcoverage_percent'] for row in selected_coverage_rows),
        'base_median_solve_time_s': float(np.median([
            row['base_solve_time_s'] for row in successful])),
        'pinn_median_solve_time_s': float(np.median([
            row['pinn_solve_time_s'] for row in successful])),
        'true_median_solve_time_s': float(np.median([
            row['true_solve_time_s'] for row in successful])),
        'true_to_pinn_median_solver_speedup': float(
            np.median([row['true_solve_time_s'] for row in successful])
            / np.median([row['pinn_solve_time_s'] for row in successful])
        ),
        'relative_cost_gap_min_percent': min(
            row['pinn_true_relative_cost_gap_percent'] for row in successful),
        'relative_cost_gap_max_percent': max(
            row['pinn_true_relative_cost_gap_percent'] for row in successful),
        'value_retention_min_percent': min(
            row['value_retention_percent'] for row in successful),
        'value_retention_max_percent': max(
            row['value_retention_percent'] for row in successful),
        'pg_l1_min_mw': min(row['pinn_true_pg_l1_mw'] for row in successful),
        'pg_l1_max_mw': max(row['pinn_true_pg_l1_mw'] for row in successful),
        'base_max_tso_branch_utilization_percent': max(
            row['base_max_tso_branch_utilization_percent'] for row in successful),
        'pinn_max_tso_branch_utilization_percent': max(
            row['pinn_max_tso_branch_utilization_percent'] for row in successful),
        'true_max_tso_branch_utilization_percent': max(
            row['true_max_tso_branch_utilization_percent'] for row in successful),
        'tso_ge99_utilization_scenario_count': sum(
            max(row['base_tso_branch_ge99_utilization_ratio_percent'],
                row['pinn_tso_branch_ge99_utilization_ratio_percent'],
                row['true_tso_branch_ge99_utilization_ratio_percent']) > 0.0
            for row in successful),
        'representative_multistart_optimization_success_count': sum(
            bool(row['optimization_success']) for row in multistart_rows),
        'representative_multistart_feasible_count': sum(
            bool(row['feasible_solution']) for row in multistart_rows),
        'representative_multistart_total_count': len(multistart_rows),
        'representative_multistart_group_count': len(multistart_summary),
        'representative_multistart_max_cost_range': (
            max(multistart_cost_ranges) if multistart_cost_ranges else np.nan),
    }
    aggregate_path = out_dir / 'reviewer_evidence_summary.csv'
    with aggregate_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate.keys()))
        writer.writeheader()
        writer.writerow(aggregate)
    return summary_path, coverage_path, selected_path, aggregate_path


def save_reviewer_evidence(coverage, coverage_raw, convergence,
                           scenario_rows, multistart_rows):
    out_dir = OUT_ROOT / 'load_scenarios'
    with (out_dir / 'coverage_summary.csv').open(
            'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage.keys()))
        writer.writeheader()
        writer.writerow(coverage)
    with (out_dir / 'coverage_convergence.csv').open(
            'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(convergence[0].keys()))
        writer.writeheader()
        writer.writerows(convergence)
    np.savez_compressed(
        out_dir / 'coverage_raw.npz',
        **coverage_raw,
        definition=np.asarray(
            'same directional metric as approximate_polygon_coverage.py: '
            'rho=k_pinn/k_true; coverage=mean(rho); '
            'under=mean(max(1-rho,0)); over=mean(max(rho-1,0))'
        ),
    )
    with (out_dir / 'true_multistart.csv').open(
            'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(multistart_rows[0].keys()))
        writer.writeheader()
        writer.writerows(multistart_rows)

    successful = [row for row in multistart_rows if row['success']]
    costs = np.asarray([row['cost'] for row in successful], dtype=float)
    pcc = np.asarray([[row['pcc_p_pu_dsbase'], row['pcc_q_pu_dsbase']]
                      for row in successful], dtype=float)
    nominal = min(scenario_rows, key=lambda row: abs(row['ts_load_factor'] - 1.0))
    combined = {
        **coverage,
        **nominal,
        'true_multistart_success_count': len(successful),
        'true_multistart_total_count': len(multistart_rows),
        'true_multistart_cost_range': (
            float(costs.max() - costs.min()) if costs.size else np.nan
        ),
        'true_multistart_max_pcc_distance_pu': (
            float(np.max(np.linalg.norm(pcc - pcc[0], axis=1)))
            if len(pcc) else np.nan
        ),
    }
    combined_path = out_dir / 'reviewer_evidence_table.csv'
    with combined_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(combined.keys()))
        writer.writeheader()
        writer.writerow(combined)
    return combined_path


def run_load_scenarios():
    global DIAGNOSTIC_CURRENT_MULTIPLIER, TS_LOAD_FACTOR, USE_DSO_CURRENT_LIMITS, OUT
    DIAGNOSTIC_CURRENT_MULTIPLIER = 2.0
    USE_DSO_CURRENT_LIMITS = False
    print('=== BaseFeasibility pre-check ===')
    feasible_factors, precheck_path = precheck_load_factors()
    if not feasible_factors:
        raise RuntimeError('allTSOload scenarioBaseNone of the models are feasible')

    print('\n=== case33Nominal operating condition coverage calculation ===')
    coverage_dsppc = getattr(TD_case, DS_CASE)(root_voltage=PCC_VOLTAGE)
    coverage_pinn = load_nominal_pinn(coverage_dsppc)
    coverage, coverage_raw, convergence = compute_nominal_coverage(
        coverage_dsppc, coverage_pinn
    )
    print(f"coverage={coverage['coverage_percent']:.4f}% "
          f"under={coverage['undercoverage_percent']:.4f}% "
          f"over={coverage['overcoverage_percent']:.4f}%")

    scenario_rows = []
    for factor in feasible_factors:
        TS_LOAD_FACTOR = float(factor)
        OUT = (
            OUT_ROOT / 'load_scenarios'
            / f'load_factor_{factor:.2f}'
        )
        print('\n' + '=' * 72)
        print(f'TSOload scenario: alpha={factor:.2f}, DSOWireless line capacity constraints')
        print('=' * 72)
        row = run_single_scenario()
        row.update(coverage)
        scenario_rows.append(row)

    OUT = OUT_ROOT
    print('\n=== TrueModel 5 initial value check (alpha=1.00) ===')
    multistart_rows = run_true_multistart()
    summary_path = save_load_scenario_summary(scenario_rows)
    evidence_path = save_reviewer_evidence(
        coverage, coverage_raw, convergence, scenario_rows, multistart_rows
    )
    print('\n=== MuchTSOSummary of load scenarios ===')
    for row in scenario_rows:
        print(
            f"alpha={row['ts_load_factor']:.2f} "
            f"base={row['base_cost']:.6f} "
            f"pinn={row['pinn_cost']:.6f} "
            f"true={row['true_cost']:.6f} "
            f"gap={row['pinn_true_cost_gap']:.6f} "
            f"retention={row['value_retention_percent']:.3f}% "
            f"Pg_L1={row['pinn_true_pg_l1_mw']:.5f} MW"
        )
    print(f'Basepre-check: {precheck_path}')
    print(f'Scenario summary: {summary_path}')
    print(f'Review evidence core table: {evidence_path}')


def run_operating_grid():
    """Calculate first50aDSOcoverage, then5Each quantile represents the operating condition5x5Scheduling."""
    global DIAGNOSTIC_CURRENT_MULTIPLIER, TS_LOAD_FACTOR, USE_DSO_CURRENT_LIMITS, OUT
    DIAGNOSTIC_CURRENT_MULTIPLIER = 2.0
    USE_DSO_CURRENT_LIMITS = False
    dsppc_nominal = getattr(TD_case, DS_CASE)(root_voltage=PCC_VOLTAGE)
    probe = load_nominal_pinn(dsppc_nominal)
    tsppc_original = getattr(TD_case, TS_CASE)()
    nominal_x_ref = nominal_reference_point(dsppc_nominal, probe)
    # Decompose the original TSO load exactly once. Every DSO condition then
    # sees the same native host load, so DSO-load changes are not cancelled.
    nominal_host_tsppc = make_host_ts_case(
        tsppc_original, dsppc_nominal, nominal_x_ref)
    paper_dthetas = generate_paper_dtheta_samples(probe['dim_theta'])
    scenario_rows = []
    coverage_rows = []
    coverage_raw_by_dso = {}

    print('\n=== stage1: Reproduction paper50aDSOWorking conditionsx100direction coverage ===')
    for sample_index in range(PAPER_N_DTHETA):
        dtheta = paper_dthetas[sample_index]
        pinn_data = predict_pinn(probe, dtheta)
        dsppc = apply_dtheta_to_dsppc(dsppc_nominal, pinn_data)
        condition = dso_condition_statistics(dsppc_nominal, dsppc, dtheta)
        print('\n' + '#' * 76)
        print(f'DSOrandom sample {sample_index + 1}/{PAPER_N_DTHETA}'
              f' (Paperdthetasequence index'
              f'{sample_index}) ')
        print(f'dtheta mean/std={condition["dtheta_mean"]:+.4f}/'
              f'{condition["dtheta_std"]:.4f}, P/QTotal load ratio='
              f'{condition["dso_active_load_factor"]:.4f}/'
              f'{condition["dso_reactive_load_factor"]:.4f}')
        print('#' * 76)
        coverage, coverage_raw, _ = compute_nominal_coverage(
            dsppc_nominal, pinn_data,
            n_dirs=PAPER_N_DIRS, direction_seed=sample_index,
        )
        coverage_row = {
            'dso_scenario_index': int(sample_index),
            **condition,
            **coverage,
        }
        coverage_rows.append(coverage_row)
        coverage_raw_by_dso[sample_index] = {
            **coverage_raw,
            'dtheta': np.asarray(dtheta, dtype=float),
            'dso_scenario_index': np.asarray(sample_index),
            'dtheta_generation_seed': np.asarray(PAPER_DTHETA_SEED),
            'direction_seed': np.asarray(sample_index),
        }
        print(f"coverage={coverage['coverage_percent']:.4f}% "
              f"under={coverage['undercoverage_percent']:.4f}% "
              f"over={coverage['overcoverage_percent']:.4f}%")

    selected_coverage_rows = select_coverage_percentile_conditions(coverage_rows)
    selection_by_index = {
        row['dso_scenario_index']: row for row in selected_coverage_rows
    }
    for row in coverage_rows:
        selected = selection_by_index.get(row['dso_scenario_index'])
        row['selected_for_downstream'] = selected is not None
        row['coverage_percentile'] = (
            selected['coverage_percentile'] if selected is not None else np.nan
        )
        row['coverage_rank_zero_based'] = (
            selected['coverage_rank_zero_based'] if selected is not None else np.nan
        )

    print('\n=== Coverage quantiles represent operating conditions ===')
    for selected in selected_coverage_rows:
        print(
            f"P{selected['coverage_percentile']:02d}: "
            f"DSO sample={selected['dso_scenario_index'] + 1}, "
            f"coverage={selected['coverage_percent']:.4f}%"
        )

    print('\n=== stage2: 5Coverage quantile operating conditionsx5aTSOload level ===')
    for selected in selected_coverage_rows:
        sample_index = int(selected['dso_scenario_index'])
        percentile = int(selected['coverage_percentile'])
        dtheta = paper_dthetas[sample_index]
        pinn_data = predict_pinn(probe, dtheta)
        for ts_factor in GRID_TS_LOAD_FACTORS:
            TS_LOAD_FACTOR = float(ts_factor)
            OUT = (
                OUT_ROOT / GRID_OUT_DIRNAME
                / f'P{percentile:02d}_dso_sample_{sample_index + 1:02d}'
                / f'tso_{ts_factor:.2f}'
            )
            print('\n' + '=' * 76)
            print(f'P{percentile:02d} DSO sample={sample_index + 1} '
                  f'x TSO={ts_factor:.2f}')
            print('=' * 76)
            row = run_single_scenario(
                dso_dtheta_level=np.nan,
                pinn_data=pinn_data,
                dsppc_nominal=dsppc_nominal,
                dso_scenario_index=sample_index,
                nominal_host_tsppc=nominal_host_tsppc,
            )
            row.update(selected)
            scenario_rows.append(row)

    OUT = OUT_ROOT
    print('\n=== 3a representative sceneBase/PINN/TrueFive initial value checks ===')
    representative_pairs = (
        (selected_coverage_rows[0], GRID_TS_LOAD_FACTORS[0]),
        (selected_coverage_rows[len(selected_coverage_rows) // 2],
         GRID_TS_LOAD_FACTORS[len(GRID_TS_LOAD_FACTORS) // 2]),
        (selected_coverage_rows[-1], GRID_TS_LOAD_FACTORS[-1]),
    )
    multistart_rows = []
    for selected, ts_factor in representative_pairs:
        sample_index = int(selected['dso_scenario_index'])
        percentile = int(selected['coverage_percentile'])
        dtheta = paper_dthetas[sample_index]
        pinn_data = predict_pinn(probe, dtheta)
        dsppc = apply_dtheta_to_dsppc(dsppc_nominal, pinn_data)
        x_ref = nominal_reference_point(dsppc_nominal, pinn_data)
        host = scale_host_ts_load(nominal_host_tsppc, ts_factor)
        scenario_name = f'P{percentile:02d}_TSO_{ts_factor:.2f}'
        multistart_rows.extend(run_representative_multistart(
            scenario_name=scenario_name,
            host_tsppc=host,
            dsppc=dsppc,
            pinn_data=pinn_data,
            x_ref=x_ref,
            metadata={
                'coverage_percentile': percentile,
                'dso_scenario_index': sample_index,
                'coverage_percent': selected['coverage_percent'],
                'ts_load_factor': float(ts_factor),
            },
        ))
    summary_path, coverage_path, selected_path, aggregate_path = save_operating_grid(
        scenario_rows, coverage_rows, selected_coverage_rows,
        coverage_raw_by_dso, multistart_rows
    )
    print('\n=== 5x5Summary of operating conditions ===')
    print(f'success scenario: {sum(np.isfinite(row["true_cost"]) for row in scenario_rows)}'
          f'/{len(scenario_rows)}')
    print(f'Cost order is correct: {sum(row["cost_order_ok"] for row in scenario_rows)}'
          f'/{len(scenario_rows)}')
    print(f'Grid summary: {summary_path}')
    print(f'50aDSOCoverage: {coverage_path}')
    print(f'5Each quantile represents the operating condition: {selected_path}')
    print(f'Summary of review evidence: {aggregate_path}')


def main():
    ipopt = prepare_ipopt()
    print(f'Ipopt: {ipopt["executable"]}')
    print(f'Ipoptversion: {ipopt["version"]}')
    parser = argparse.ArgumentParser(description='R1.5/R3.2 downstream experiment')
    parser.add_argument(
        '--mode',
        choices=('operating-grid', 'load-scenarios', 'current-sensitivity'),
        default='operating-grid',
    )
    args = parser.parse_args()
    if args.mode == 'current-sensitivity':
        run_current_sensitivity()
    elif args.mode == 'load-scenarios':
        run_load_scenarios()
    else:
        run_operating_grid()


if __name__ == '__main__':
    main()
