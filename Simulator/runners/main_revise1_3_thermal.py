# -*- coding: utf-8 -*-
"""Run the R1.3 case33 thermal-constraint stress test."""
import argparse
import contextlib
import copy
import csv
import io
import os
import time
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from Simulator import PROJECT_ROOT
from Simulator.Approximator import PreTrainNet, FullNet, Trainer, compute_loss
from Simulator.cases import TD_case
from Simulator.reference_point import (
    REFERENCE_MEMBERSHIP_TOL,
    ReferencePointCalculator,
    reference_point_membership,
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CASENAME = 'case33bw_ds'

# PyCharm direct Run The complete experiment is executed by default; command line parameters take precedence.
SCENARIO = 'both'                # both / baseline / thermal
STAGE = 'all'                    # preview / calibrate / all / pretrain / fullnet
CURRENT_MARGIN = 1.20

# Thermal-limit prescreening compares all multipliers on identical operating points and directions without retraining.
CALIBRATION_MARGINS = (1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50)
N_CALIBRATION_DTHETA = 50
N_CALIBRATION_DIRS = 8
CALIBRATION_SEED = 42
MIN_DIRECTION_SUCCESS_RATE = 0.995
REFERENCE_CURRENT_TOL = 1e-6
N_PREVIEW_DIRS = 100

# with original FullNet Structure and training scale remain consistent.
N_TRAIN_PRE = 500                # actual 4*N = 2000
N_TRAIN_FULL = 20                # phase1=79, phase2=40
HIDDEN_SIZES = [128]
ACTIVATION = 'relu'
TRAIN_SEED = 0

# Formal evaluation uses the nominal point plus nine random points, each tested in 100 directions.
N_TEST_DTHETA = 10
N_DIRS_EVAL = 100
DTHETA_RANGE = (-0.5, 0.5)
ACTIVE_UTILIZATION_TOL = 0.99
OUT_ROOT = (PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' /
            'comparison' / 'thermal' / CASENAME)


def scenario_tag(scenario, margin):
    if scenario == 'baseline':
        return 'baseline'
    return f"thermal_{int(round(100.0 * margin)):03d}pct"


def scenario_dir(scenario, margin):
    return OUT_ROOT / scenario_tag(scenario, margin)


def build_case(scenario, margin):
    """Construct this experimental physics example; default original manuscript case not be modified."""
    ppc = TD_case.case33bw_ds()
    if scenario == 'thermal':
        ppc = TD_case.add_power_flow_thermal_limits(
            ppc, current_margin=margin)
    elif scenario != 'baseline':
        raise ValueError(f'Unknown scene: {scenario!r}')
    return ppc


def save_thermal_ratings(ppc, out_dir):
    """Save branch-by-branch calibration values for paper tracing and inspection."""
    if 'branch_I2max_pu2' not in ppc:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    branch = np.asarray(ppc['branch'])
    nominal = np.asarray(ppc['branch_I_nominal_pu'])
    limit = np.asarray(ppc['branch_Imax_pu'])
    i2_limit = np.asarray(ppc['branch_I2max_pu2'])
    with open(out_dir / 'thermal_ratings.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'line_index', 'from_bus', 'to_bus', 'I_nominal_pu',
            'Imax_pu', 'I2max_pu2', 'current_margin'])
        for line in range(len(branch)):
            writer.writerow([
                line, int(branch[line, 0]), int(branch[line, 1]),
                nominal[line], limit[line], i2_limit[line],
                ppc['thermal_current_margin']])
    np.savez(
        out_dir / 'thermal_ratings.npz',
        branch_from=branch[:, 0].astype(int),
        branch_to=branch[:, 1].astype(int),
        I_nominal_pu=nominal,
        Imax_pu=limit,
        I2max_pu2=i2_limit,
        current_margin=np.asarray(ppc['thermal_current_margin']),
        source=np.asarray(ppc['thermal_limit_source']),
        nominal_pf_min_voltage_pu=np.asarray(
            ppc['thermal_pf_min_voltage_pu']))


def make_dirs(n_dirs):
    angles = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def reset_training_seed(seed=TRAIN_SEED):
    """makebaselineandthermalUse the same random sequence for future retraining."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def k_max_polytope(A, b, x_ref, direction):
    av = A @ direction
    slack = b - A @ x_ref
    valid = av > 1e-12
    if not np.any(valid):
        return np.nan
    k = np.min(slack[valid] / av[valid])
    return float(k) if np.isfinite(k) else np.nan


def sync_physical_parameters(error_calculator, dtheta, init_params_dict):
    pd_init = init_params_dict['Pd_meta']['initial_value']
    qd_init = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd_init.size
    error_calculator.update_parameters({
        'Pd_meta': pd_init + dtheta[:n_pd].reshape(pd_init.shape),
        'Qd_meta': qd_init + dtheta[n_pd:].reshape(qd_init.shape),
    })


def summarize(values):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return dict(mean=np.nan, median=np.nan, p95=np.nan, count=0)
    return dict(
        mean=float(np.mean(valid)),
        median=float(np.median(valid)),
        p95=float(np.percentile(valid, 95)),
        count=int(valid.size),
    )


def evaluate_model(error_calculator, model, ppc, init_params_dict):
    """Coverage, squared error and true feasible region line utilization are saved item by item in the same direction.."""
    ec = error_calculator.copy()
    eval_dirs = make_dirs(N_DIRS_EVAL)
    ec.attach_radial(ppc, init_params_dict, n_dirs=N_DIRS_EVAL, seed=123)

    pd_init = init_params_dict['Pd_meta']['initial_value']
    qd_init = init_params_dict['Qd_meta']['initial_value']
    total = pd_init.size + qd_init.size
    rng = np.random.RandomState(42)
    test_dthetas = rng.uniform(
        DTHETA_RANGE[0], DTHETA_RANGE[1],
        size=(N_TEST_DTHETA, total))
    test_dthetas[0] = 0.0       # The first working case is specifically used to draw the nominal flexible domain comparison

    shape = (N_TEST_DTHETA, N_DIRS_EVAL)
    feasibility = np.full(shape, np.nan)
    optimality = np.full(shape, np.nan)
    coverage = np.full(shape, np.nan)
    feasibility_success = np.zeros(shape, dtype=bool)
    optimality_success = np.zeros(shape, dtype=bool)
    true_support_points = np.full((N_TEST_DTHETA, N_DIRS_EVAL, 2), np.nan)
    max_line_utilization = np.full(shape, np.nan)
    ge99_line_ratio = np.full(shape, np.nan)
    active_line_count = np.zeros(shape, dtype=int)
    active_line_mask = np.zeros(
        (N_TEST_DTHETA, N_DIRS_EVAL, len(ppc['branch'])), dtype=bool)
    membership_failure = np.zeros(N_TEST_DTHETA, dtype=bool)
    predicted_A = np.full((N_TEST_DTHETA, 36, 2), np.nan)
    predicted_b = np.full((N_TEST_DTHETA, 36), np.nan)

    i2_limit = ppc.get('branch_I2max_pu2')
    if i2_limit is not None:
        i2_limit = np.asarray(i2_limit, dtype=float)

    for idx, dtheta in enumerate(test_dthetas):
        sync_physical_parameters(ec, dtheta, init_params_dict)
        with torch.no_grad():
            dt = torch.tensor(dtheta, dtype=torch.float32, device=DEVICE)
            A, b = model(dt)
            A_np = A[0].detach().cpu().numpy()
            b_np = b[0].detach().cpu().numpy()
        predicted_A[idx] = A_np
        predicted_b[idx] = b_np
        ec.update_polytope(A_hat=A_np, b_hat=b_np)

        # The real support point is only found once, and both error directions are reused. Read immediately after solving I2.
        true_support = []
        for j, direction in enumerate(eval_dirs):
            point = ec.optimize_direction(direction, in_approx=False)
            true_support.append(point)
            if point is not None:
                true_support_points[idx, j] = point
                if i2_limit is not None:
                    i2_value = np.array([
                        pyo.value(ec.original_model.I2[line])
                        for line in ec.original_model.LINE], dtype=float)
                    utilization = np.sqrt(
                        np.maximum(i2_value, 0.0) / i2_limit)
                    max_line_utilization[idx, j] = np.max(utilization)
                    active = utilization >= ACTIVE_UTILIZATION_TOL
                    active_line_mask[idx, j] = active
                    active_line_count[idx, j] = int(np.sum(active))
                    ge99_line_ratio[idx, j] = float(np.mean(active))

        for j, direction in enumerate(eval_dirs):
            x_poly = ec.optimize_direction(direction, in_approx=True)
            x_true_proj = ec.project(x_poly) if x_poly is not None else None
            if x_poly is not None and x_true_proj is not None:
                feasibility[idx, j] = np.sum((x_poly - x_true_proj) ** 2)
                feasibility_success[idx, j] = True

            x_true = true_support[j]
            x_poly_proj = (ec.project(x_true, to_approx=True)
                           if x_true is not None else None)
            if x_true is not None and x_poly_proj is not None:
                optimality[idx, j] = np.sum((x_true - x_poly_proj) ** 2)
                optimality_success[idx, j] = True

        radial = ec.calculate_radial(eval_dirs, dtheta)
        if radial is not None:
            x_ref = radial[0]['x_ref']
            is_member, _ = reference_point_membership(
                A_np, b_np, x_ref, REFERENCE_MEMBERSHIP_TOL)
            membership_failure[idx] = not is_member
            if is_member:
                for j, (record, direction) in enumerate(zip(radial, eval_dirs)):
                    k_omega = float(record['k_omega'])
                    k_poly = k_max_polytope(
                        A_np, b_np, x_ref, direction)
                    if k_omega > 1e-9 and np.isfinite(k_poly):
                        coverage[idx, j] = k_poly / k_omega

        print(
            f"  Working conditions {idx + 1}/{N_TEST_DTHETA}: "
            f"feas {feasibility_success[idx].sum()}/{N_DIRS_EVAL}, "
            f"opt {optimality_success[idx].sum()}/{N_DIRS_EVAL}, "
            f"rho {np.isfinite(coverage[idx]).sum()}/{N_DIRS_EVAL}")

    return {
        'test_dthetas': test_dthetas,
        'eval_dirs': eval_dirs,
        'feasibility': feasibility,
        'optimality': optimality,
        'coverage': coverage,
        'feasibility_success': feasibility_success,
        'optimality_success': optimality_success,
        'true_support_points': true_support_points,
        'max_line_utilization': max_line_utilization,
        'ge99_line_ratio': ge99_line_ratio,
        'active_line_count': active_line_count,
        'active_line_mask': active_line_mask,
        'membership_failure': membership_failure,
        'predicted_A': predicted_A,
        'predicted_b': predicted_b,
    }


def ordered_boundary(points, *aligned):
    """Delete failure points and sort by relative centroid polar angle; synchronously rearrange additional arrays."""
    points = np.asarray(points, dtype=float)
    valid = np.all(np.isfinite(points), axis=1)
    clean = points[valid]
    if len(clean) < 3:
        return (clean,) + tuple(np.asarray(x)[valid] for x in aligned)
    center = np.mean(clean, axis=0)
    order = np.argsort(np.arctan2(
        clean[:, 1]-center[1], clean[:, 0]-center[0]))
    return ((clean[order],)
            + tuple(np.asarray(x)[valid][order] for x in aligned))


def polygon_area(points):
    """Shoelace formula area of ordered two-dimensional boundary points."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.nan
    return 0.5*abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1)))


def evaluation_summary_row(evaluation, scenario, margin, total_time=np.nan):
    """A formal summary is uniformly generated from successive original evaluation values, compatible with earlyNPZField."""
    feas = summarize(evaluation['feasibility'])
    opt = summarize(evaluation['optimality'])
    rho = summarize(evaluation['coverage'])
    util = summarize(evaluation['max_line_utilization'])

    true_points = np.asarray(evaluation['true_support_points'], dtype=float)
    true_success = np.all(np.isfinite(true_points), axis=-1)
    nominal_points = true_points[0][true_success[0]]
    nominal_area = polygon_area(ordered_boundary(nominal_points)[0])

    line_ratios = np.full(true_success.shape, np.nan, dtype=float)
    if scenario == 'thermal':
        if 'ge99_line_ratio' in evaluation:
            stored = np.asarray(evaluation['ge99_line_ratio'], dtype=float)
            line_ratios[true_success] = stored[true_success]
        elif 'active_line_mask' in evaluation:
            active_mask = np.asarray(evaluation['active_line_mask'], dtype=bool)
            line_ratios[true_success] = np.mean(
                active_mask[true_success], axis=1)
    valid_ratio = line_ratios[np.isfinite(line_ratios)]
    if valid_ratio.size:
        ratio_mean = float(np.mean(valid_ratio))
        ratio_median = float(np.median(valid_ratio))
        ratio_p95 = float(np.percentile(valid_ratio, 95))
        any_active_rate = float(np.mean(valid_ratio > 0.0))
    else:
        ratio_mean = ratio_median = ratio_p95 = any_active_rate = np.nan

    feasibility_success = np.asarray(
        evaluation.get('feasibility_success', np.isfinite(
            evaluation['feasibility'])), dtype=bool)
    optimality_success = np.asarray(
        evaluation.get('optimality_success', np.isfinite(
            evaluation['optimality'])), dtype=bool)

    return {
        'scenario': scenario_tag(scenario, margin),
        'current_margin': margin if scenario == 'thermal' else np.nan,
        'feas_mean': feas['mean'],
        'feas_median': feas['median'],
        'feas_p95': feas['p95'],
        'opt_mean': opt['mean'],
        'opt_median': opt['median'],
        'opt_p95': opt['p95'],
        'coverage_mean': rho['mean'],
        'coverage_median': rho['median'],
        'coverage_p95': rho['p95'],
        'coverage_count': rho['count'],
        'membership_failed_cases': int(np.sum(
            evaluation['membership_failure'])),
        'true_support_success_rate': float(np.mean(true_success)),
        'feasibility_success_rate': float(np.mean(feasibility_success)),
        'optimality_success_rate': float(np.mean(optimality_success)),
        'nominal_true_region_area': nominal_area,
        'max_utilization_mean': util['mean'],
        'max_utilization_p95': util['p95'],
        'any_ge99_direction_rate': any_active_rate,
        'ge99_line_ratio_mean': ratio_mean,
        'ge99_line_ratio_median': ratio_median,
        'ge99_line_ratio_p95': ratio_p95,
        'fullnet_total_time': float(total_time),
    }


def solve_nominal_true_boundary(scenario, margin, directions):
    """Find the support point in the direction of the true feasible region under nominal operating conditions; thermalAdditional read line utilization."""
    ppc = build_case(scenario, margin)
    case = TD_case.DScase_train(
        casedata=ppc, model_type='fullnet', plot_flag=False, device=DEVICE)
    ec = case['errorcalculator'].copy()
    dtheta = np.zeros(case['params']['count'], dtype=float)
    sync_physical_parameters(ec, dtheta, case['params']['params_dict'])
    points = np.full((len(directions), 2), np.nan)
    success = np.zeros(len(directions), dtype=bool)
    max_utilization = np.full(len(directions), np.nan)
    active = np.zeros(len(directions), dtype=bool)
    i2_limit = (np.asarray(ppc['branch_I2max_pu2'], dtype=float)
                if scenario == 'thermal' else None)

    for idx, direction in enumerate(directions):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            point = ec.optimize_direction(direction, in_approx=False)
        if point is not None and np.all(np.isfinite(point)):
            points[idx] = point
            success[idx] = True
            if i2_limit is not None:
                i2 = np.array([
                    pyo.value(ec.original_model.I2[line])
                    for line in ec.original_model.LINE], dtype=float)
                utilization = np.sqrt(np.maximum(i2, 0.0)/i2_limit)
                max_utilization[idx] = float(np.max(utilization))
                active[idx] = max_utilization[idx] >= ACTIVE_UTILIZATION_TOL
        if ((idx+1) % 10 == 0 or idx+1 == len(directions)):
            print(f'  [{scenario}] direction {idx+1}/{len(directions)}, '
                  f'success {success[:idx+1].sum()}/{idx+1}')
    return ppc, case, points, success, max_utilization, active


def run_true_region_preview(margin=CURRENT_MARGIN, n_dirs=N_PREVIEW_DIRS):
    """Compare nominal operating conditions baseline/thermal Real domain and automatically generate graphs."""
    if not np.isfinite(margin) or margin <= 1.0:
        raise ValueError('previewof--marginmust be greater than1')
    if n_dirs < 8:
        raise ValueError('previewThe number of directions is at least8')
    out = OUT_ROOT / f'preview_thermal_{int(round(100*margin)):03d}pct'
    out.mkdir(parents=True, exist_ok=True)
    directions = make_dirs(n_dirs)
    print(f'\n[preview] Nominal operating condition true feasible region: baseline vs thermal, '
          f'kappa={margin:g}, Number of directions={n_dirs}')

    base_ppc, base_case, base_raw, base_success, _, _ = \
        solve_nominal_true_boundary('baseline', margin, directions)
    thermal_ppc, _, thermal_raw, thermal_success, thermal_util, thermal_active = \
        solve_nominal_true_boundary('thermal', margin, directions)
    save_thermal_ratings(thermal_ppc, out)

    base_ordered, = ordered_boundary(base_raw)
    thermal_ordered, thermal_util_ordered, thermal_active_ordered = \
        ordered_boundary(thermal_raw, thermal_util, thermal_active)
    area_base = polygon_area(base_ordered)
    area_thermal = polygon_area(thermal_ordered)
    area_reduction = (
        (area_base-area_thermal)/area_base
        if np.isfinite(area_base) and area_base > 0 and np.isfinite(area_thermal)
        else np.nan)

    reference = ReferencePointCalculator(
        ppc=base_ppc,
        init_params_dict=base_case['params']['params_dict'],
        original_model=base_case['errorcalculator'].original_model,
        solver='ipopt').compute(np.zeros(base_case['params']['count']))
    reference = (np.asarray(reference, dtype=float) if reference is not None
                 else np.full(2, np.nan))

    data_path = out / 'true_region_preview_data.npz'
    np.savez(
        data_path,
        casename=np.asarray(CASENAME), margin=np.asarray(margin),
        n_directions=np.asarray(n_dirs), directions=directions,
        baseline_points_raw=base_raw, thermal_points_raw=thermal_raw,
        baseline_success=base_success, thermal_success=thermal_success,
        baseline_boundary=base_ordered, thermal_boundary=thermal_ordered,
        thermal_max_utilization_raw=thermal_util,
        thermal_max_utilization=thermal_util_ordered,
        thermal_active_raw=thermal_active,
        thermal_active=thermal_active_ordered,
        active_utilization_tol=np.asarray(ACTIVE_UTILIZATION_TOL),
        reference_point=reference,
        baseline_area=np.asarray(area_base), thermal_area=np.asarray(area_thermal),
        area_reduction=np.asarray(area_reduction),
        boundary_definition=np.asarray(
            'directional support-point outline of the true nonlinear model'))

    print(f'[preview] baselinesuccess={base_success.sum()}/{n_dirs}, '
          f'thermalsuccess={thermal_success.sum()}/{n_dirs}')
    print(f'[preview] area baseline={area_base:.6g}, thermal={area_thermal:.6g}, '
          f'shrink={100*area_reduction:.2f}%')
    active_rate = (float(np.mean(thermal_active[thermal_success]))
                   if np.any(thermal_success) else np.nan)
    print('[preview] At least one line utilization>=99%The boundary direction ratio of='
          f'{100*active_rate:.1f}%')
    print(f'[preview] raw data: {data_path}')

    from Simulator.draw_pictures.R1_3_thermal_region_preview_plot import \
        plot_preview
    plot_preview(data_path)
    return data_path


def parse_margin_list(text):
    """parse ``1.10,1.15,...``, Remove duplicates and sort them from small to large."""
    try:
        values = sorted({float(x.strip()) for x in text.split(',') if x.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError('The multiplier must be a comma separated number') from exc
    if not values or any((not np.isfinite(x) or x <= 1.0) for x in values):
        raise argparse.ArgumentTypeError('All heat capacity ratios must be greater than1finite number of')
    return tuple(values)


def pypower_currents_for_dtheta(ppc, dtheta, init_params_dict):
    """Calculate an inflexibility operating pointPYPOWERSending end branch current (p.u.) ."""
    from pypower.api import ppoption, runpf
    from pypower.idx_brch import F_BUS, PF, QF
    from pypower.idx_bus import BUS_I, VM

    pd0 = init_params_dict['Pd_meta']['initial_value']
    qd0 = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd0.size
    pd_meta = pd0 + dtheta[:n_pd].reshape(pd0.shape)
    qd_meta = qd0 + dtheta[n_pd:].reshape(qd0.shape)

    pf_case = copy.deepcopy(ppc)
    pf_case['bus'] = np.asarray(pf_case['bus'], dtype=float).copy()
    pf_case['branch'] = np.asarray(pf_case['branch'], dtype=float).copy()
    pf_case['gen'] = np.asarray(pf_case['gen'], dtype=float).copy()
    base_mva = float(pf_case['baseMVA'])
    pd_nominal = np.asarray(ppc['bus'][:, 2], dtype=float)
    qd_nominal = np.asarray(ppc['bus'][:, 3], dtype=float)
    pf_case['bus'][:, 2] = (0.8 + 0.4*pd_meta.reshape(-1)) * pd_nominal
    pf_case['bus'][:, 3] = (0.8 + 0.4*qd_meta.reshape(-1)) * qd_nominal

    # PYPOWEROccasionally output prompts; only presssuccessand numerical recording without refreshing the screen.
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        try:
            result, success = runpf(
                pf_case, ppoption(VERBOSE=0, OUT_ALL=0))
        except Exception:
            return None
    if not success:
        return None
    branch = np.asarray(result['branch'], dtype=float)
    bus = np.asarray(result['bus'], dtype=float)
    voltage = {int(row[BUS_I]): float(row[VM]) for row in bus}
    currents = np.array([
        np.hypot(row[PF], row[QF]) / base_mva / voltage[int(row[F_BUS])]
        for row in branch], dtype=float)
    if (currents.shape != (len(ppc['branch']),)
            or not np.all(np.isfinite(currents))):
        return None
    return currents


def run_margin_calibration(margins, n_dtheta=N_CALIBRATION_DTHETA,
                           n_dirs=N_CALIBRATION_DIRS,
                           min_success=MIN_DIRECTION_SUCCESS_RATE):
    """Without training the network, screen the minimum heat capacity rate that is “feasible and will activate“."""
    if n_dtheta <= 0 or n_dirs <= 0:
        raise ValueError('The number of calibration conditions and the number of directions must be positive integers')
    if not (0.0 < min_success <= 1.0):
        raise ValueError('The direction solution success rate threshold must be at(0,1]')

    out = OUT_ROOT / 'calibration'
    out.mkdir(parents=True, exist_ok=True)
    base_ppc = TD_case.case33bw_ds()
    # Only build the original model once to obtain parameter definitions that are completely consistent with training.
    base_case = TD_case.DScase_train(
        casedata=base_ppc, model_type='fullnet', plot_flag=False,
        device=DEVICE)
    init_params = base_case['params']['params_dict']
    total_params = base_case['params']['count']
    rng = np.random.RandomState(CALIBRATION_SEED)
    dthetas = rng.uniform(
        DTHETA_RANGE[0], DTHETA_RANGE[1],
        size=(n_dtheta, total_params))
    dthetas[0] = 0.0
    directions = make_dirs(n_dirs)

    # The no-flexibility power flow is independent of the multiplier and is solved only once;
    # each multiplier is evaluated with a different Imax value.
    reference_currents = np.full(
        (n_dtheta, len(base_ppc['branch'])), np.nan)
    reference_pf_success_once = np.zeros(n_dtheta, dtype=bool)
    for condition_idx, dtheta in enumerate(dthetas):
        currents = pypower_currents_for_dtheta(base_ppc, dtheta, init_params)
        if currents is not None:
            reference_currents[condition_idx] = currents
            reference_pf_success_once[condition_idx] = True

    n_margin = len(margins)
    pf_success = np.zeros((n_margin, n_dtheta), dtype=bool)
    reference_feasible = np.zeros((n_margin, n_dtheta), dtype=bool)
    reference_max_utilization = np.full((n_margin, n_dtheta), np.nan)
    direction_success = np.zeros((n_margin, n_dtheta, n_dirs), dtype=bool)
    direction_max_utilization = np.full(
        (n_margin, n_dtheta, n_dirs), np.nan)
    direction_active = np.zeros((n_margin, n_dtheta, n_dirs), dtype=bool)
    rows = []

    print('\n[calibrate] Only filter magnification, no trainingPreTrainNet/FullNet')
    print(f'[calibrate] {n_dtheta}operating conditions × {n_dirs}direction × '
          f'{n_margin}magnification')
    for margin_idx, margin in enumerate(margins):
        start = time.time()
        ppc = build_case('thermal', margin)
        current_limit = np.asarray(ppc['branch_Imax_pu'], dtype=float)

        # No flexibility reference point check: the reference point used for coverage must be within the allowed range of thermal constraints.
        for condition_idx in range(n_dtheta):
            if not reference_pf_success_once[condition_idx]:
                continue
            pf_success[margin_idx, condition_idx] = True
            currents = reference_currents[condition_idx]
            utilization = currents / current_limit
            reference_max_utilization[margin_idx, condition_idx] = np.max(
                utilization)
            reference_feasible[margin_idx, condition_idx] = bool(
                np.all(utilization <= 1.0 + REFERENCE_CURRENT_TOL))

        # Skip directional screening when the reference operating point is already infeasible.
        reference_pass = bool(
            np.all(pf_success[margin_idx])
            and np.all(reference_feasible[margin_idx]))
        if not reference_pass:
            print(f'  kappa={margin:.2f} Base point not achieved100%Feasible, skip the direction to solve')
        else:
            # True Flexible Domain Check: Press the same as trainingPyomomodel andIpoptFind direction support point.
            thermal_case = TD_case.DScase_train(
                casedata=ppc, model_type='fullnet', plot_flag=False,
                device=DEVICE)
            ec = thermal_case['errorcalculator'].copy()
            i2_limit = np.asarray(ppc['branch_I2max_pu2'], dtype=float)
            for condition_idx, dtheta in enumerate(dthetas):
                sync_physical_parameters(ec, dtheta, init_params)
                for direction_idx, direction in enumerate(directions):
                    # Solve silently because calibration records its own success-rate statistics.
                    with contextlib.redirect_stdout(io.StringIO()), \
                            contextlib.redirect_stderr(io.StringIO()):
                        point = ec.optimize_direction(direction, in_approx=False)
                    if point is None or not np.all(np.isfinite(point)):
                        continue
                    direction_success[
                        margin_idx, condition_idx, direction_idx] = True
                    i2 = np.array([
                        pyo.value(ec.original_model.I2[line])
                        for line in ec.original_model.LINE], dtype=float)
                    utilization = np.sqrt(np.maximum(i2, 0.0) / i2_limit)
                    max_util = float(np.max(utilization))
                    direction_max_utilization[
                        margin_idx, condition_idx, direction_idx] = max_util
                    direction_active[
                        margin_idx, condition_idx, direction_idx] = (
                            max_util >= ACTIVE_UTILIZATION_TOL)
                if ((condition_idx + 1) % 10 == 0
                        or condition_idx + 1 == n_dtheta):
                    successes = int(direction_success[
                        margin_idx, :condition_idx+1].sum())
                    attempted = (condition_idx + 1) * n_dirs
                    print(f'  kappa={margin:.2f} Working conditions '
                          f'{condition_idx+1}/{n_dtheta}, direction success '
                          f'{successes}/{attempted}')

        ref_pf_rate = float(np.mean(pf_success[margin_idx]))
        ref_feasible_rate = float(np.mean(reference_feasible[margin_idx]))
        dir_success_rate = float(np.mean(direction_success[margin_idx]))
        all_dirs_rate = float(np.mean(np.all(
            direction_success[margin_idx], axis=1)))
        valid_direction = direction_success[margin_idx]
        active_rate = (float(np.mean(direction_active[margin_idx][valid_direction]))
                       if np.any(valid_direction) else 0.0)
        valid_util = direction_max_utilization[margin_idx][valid_direction]
        util_p95 = (float(np.percentile(valid_util, 95))
                    if valid_util.size else np.nan)
        util_max = float(np.max(valid_util)) if valid_util.size else np.nan
        accepted = bool(
            ref_pf_rate == 1.0
            and ref_feasible_rate == 1.0
            and dir_success_rate >= min_success
            and active_rate > 0.0)
        row = {
            'current_margin': float(margin),
            'n_dtheta': n_dtheta,
            'n_directions': n_dirs,
            'reference_pf_success_rate': ref_pf_rate,
            'reference_thermal_feasible_rate': ref_feasible_rate,
            'direction_screen_performed': reference_pass,
            'direction_solve_success_rate': dir_success_rate,
            'all_directions_success_condition_rate': all_dirs_rate,
            'direction_with_any_line_ge99_ratio': active_rate,
            'direction_max_utilization_p95': util_p95,
            'direction_max_utilization_max': util_max,
            'accepted': accepted,
            'elapsed_seconds': time.time()-start,
        }
        rows.append(row)
        print(
            f"[kappa={margin:.2f}] reference feasible="
            f"{ref_feasible_rate:.1%}, direction success={dir_success_rate:.1%}, "
            f"any-line>=99%={active_rate:.1%}, accepted={accepted}")

    fields = list(rows[0].keys())
    csv_path = out / 'thermal_margin_screen.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        out / 'thermal_margin_screen_raw.npz',
        margins=np.asarray(margins), dthetas=dthetas, directions=directions,
        reference_currents=reference_currents,
        pf_success=pf_success, reference_feasible=reference_feasible,
        reference_max_utilization=reference_max_utilization,
        direction_success=direction_success,
        direction_max_utilization=direction_max_utilization,
        direction_active=direction_active,
        min_direction_success_rate=np.asarray(min_success),
        active_utilization_tol=np.asarray(ACTIVE_UTILIZATION_TOL),
        seed=np.asarray(CALIBRATION_SEED))

    accepted = [row for row in rows if row['accepted']]
    recommendation = min(accepted, key=lambda x: x['current_margin']) \
        if accepted else None
    recommendation_path = out / 'recommendation.txt'
    with open(recommendation_path, 'w', encoding='utf-8') as f:
        if recommendation is None:
            f.write('No magnification meets all filter criteria at the same time. please checkCSVThen expand the magnification range or adjust the calibration method..\n')
        else:
            selected = recommendation['current_margin']
            f.write(f'recommended_margin={selected:.6g}\n')
            f.write('criteria: reference PF success=100%, reference thermal '
                    f'feasible=100%, direction success>={min_success:.3%}, '
                    'directions with any line utilization>=99% >0\n')
            f.write('next_command:\n')
            f.write('python -m Simulator.runners.main_revise1_3_thermal '
                    f'--scenario thermal --stage all --margin {selected:g}\n')

    print(f'\n[calibrate] Summary: {csv_path}')
    print(f'[calibrate] original value: {out / "thermal_margin_screen_raw.npz"}')
    if recommendation is None:
        print('[calibrate] There are no qualifying magnifications among the current candidates; do not start formalthermaltraining.')
    else:
        selected = recommendation['current_margin']
        print(f'[calibrate] Recommended minimum magnification: kappa={selected:g}')
        print('[calibrate] Next run: ')
        print('  python -m Simulator.runners.main_revise1_3_thermal '
              f'--scenario thermal --stage all --margin {selected:g}')
    return rows, recommendation


def run_pretrain(scenario, margin):
    reset_training_seed()
    ppc = build_case(scenario, margin)
    out = scenario_dir(scenario, margin)
    pretrain_dir = out / 'pretrain'
    weights = pretrain_dir / 'pretrainnet_weights.pth'
    save_thermal_ratings(ppc, out)

    p_rated = np.sum(ppc['bus'][:, 2]) / ppc['baseMVA']
    case = TD_case.DScase_train(
        casedata=ppc, model_type='pretrainnet', plot_flag=False,
        device=DEVICE, figure_dir=str(pretrain_dir / 'figures'))
    model = PreTrainNet(
        case['A_hat'], case['b_hat'], is_epigraph=False, device=DEVICE)
    trainer = Trainer(
        model=model, error_calculator=case['errorcalculator'],
        compute_loss=compute_loss)
    trainer.configure(**case['trainer_configure'])
    trainer.configure(lr=1e-1 / p_rated, rate_opt_feas=0.6)
    trainer.initialize()

    print(f"[pretrain] scenario={scenario_tag(scenario, margin)} device={DEVICE}")
    start = time.time()
    trainer.train(n_train=N_TRAIN_PRE * 4, params_data=case['params'])
    elapsed = time.time() - start
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    np.savez(
        pretrain_dir / 'training_time.npz',
        pretrain_time=np.asarray(elapsed),
        scenario=np.asarray(scenario),
        current_margin=np.asarray(margin if scenario == 'thermal' else np.nan))
    print(f"[pretrain] Complete: {elapsed:.1f}s\nweight: {weights}")


def run_fullnet(scenario, margin):
    reset_training_seed()
    ppc = build_case(scenario, margin)
    out = scenario_dir(scenario, margin)
    pretrain_weights = out / 'pretrain' / 'pretrainnet_weights.pth'
    if not pretrain_weights.exists():
        raise FileNotFoundError(
            f"not found {pretrain_weights}, Please run the same first scenario/margin of pretrain")
    save_thermal_ratings(ppc, out)

    p_rated = np.sum(ppc['bus'][:, 2]) / ppc['baseMVA']
    case = TD_case.DScase_train(
        casedata=ppc, model_type='fullnet', plot_flag=False, device=DEVICE)
    init_params_dict = case['params']['params_dict']

    pre_model = PreTrainNet(
        case['A_hat'], case['b_hat'], is_epigraph=False, device=DEVICE)
    pre_model.load_state_dict(torch.load(pretrain_weights, map_location=DEVICE))
    A_pre, b_pre = pre_model()
    A_pre = A_pre[0].detach().cpu().numpy()
    b_pre = b_pre[0].detach().cpu().numpy()

    model = FullNet(
        dim_theta=case['params']['count'], A_init=A_pre, b_init=b_pre,
        hidden_sizes=HIDDEN_SIZES, activation=ACTIVATION,
        device=DEVICE).to(DEVICE)
    trainer = Trainer(
        model=model, error_calculator=case['errorcalculator'],
        compute_loss=compute_loss)
    trainer.configure(**case['trainer_configure'])

    print(f"[fullnet] scenario={scenario_tag(scenario, margin)} device={DEVICE}")
    total_start = time.time()
    trainer.configure(lr=3e-5 / p_rated, rate_opt_feas=0.6)
    trainer.initialize()
    trainer.train(n_train=N_TRAIN_FULL * 4 - 1, params_data=case['params'])
    phase1_time = time.time() - total_start

    phase2_start = time.time()
    trainer.configure(lr=1e-5 / p_rated, rate_opt_feas=1e-4)
    trainer.initialize()
    trainer.train(n_train=N_TRAIN_FULL * 2, params_data=case['params'])
    phase2_time = time.time() - phase2_start
    total_time = time.time() - total_start

    fullnet_dir = out / 'fullnet'
    fullnet_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fullnet_dir / 'fullnet_weights.pth')
    np.savez(
        fullnet_dir / 'training_time.npz',
        phase1_time=np.asarray(phase1_time),
        phase2_time=np.asarray(phase2_time),
        fullnet_total_time=np.asarray(total_time),
        scenario=np.asarray(scenario),
        current_margin=np.asarray(margin if scenario == 'thermal' else np.nan))

    print('[eval] Start unified evaluation of fixed operating conditions and fixed directions')
    evaluation = evaluate_model(
        case['errorcalculator'], model, ppc, init_params_dict)
    np.savez(
        out / 'evaluation_data.npz',
        **evaluation,
        casename=np.asarray(CASENAME),
        scenario=np.asarray(scenario),
        scenario_tag=np.asarray(scenario_tag(scenario, margin)),
        current_margin=np.asarray(margin if scenario == 'thermal' else np.nan),
        error_definition=np.asarray('squared_euclidean_distance'),
        coverage_definition=np.asarray('reference_point_radial_kP_over_kOmega'),
        active_utilization_tol=np.asarray(ACTIVE_UTILIZATION_TOL))

    row = evaluation_summary_row(
        evaluation, scenario, margin, total_time=total_time)
    save_summary(row, out)
    refresh_comparison_summary()
    print(
        f"[done] rho={row['coverage_mean']:.4f}, "
        f"feas={row['feas_mean']:.3e}, opt={row['opt_mean']:.3e}, "
        f"directions(any line>=99%)="
        f"{row['any_ge99_direction_rate']:.1%}, "
        f"mean(lines>=99%)={row['ge99_line_ratio_mean']:.1%}\n"
        f"result: {out}")


SUMMARY_FIELDS = [
    'scenario', 'current_margin',
    'feas_mean', 'feas_median', 'feas_p95',
    'opt_mean', 'opt_median', 'opt_p95',
    'coverage_mean', 'coverage_median', 'coverage_p95', 'coverage_count',
    'membership_failed_cases',
    'true_support_success_rate', 'feasibility_success_rate',
    'optimality_success_rate', 'nominal_true_region_area',
    'max_utilization_mean', 'max_utilization_p95',
    'any_ge99_direction_rate',
    'ge99_line_ratio_mean', 'ge99_line_ratio_median',
    'ge99_line_ratio_p95', 'fullnet_total_time']


def save_summary(row, out):
    with open(out / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def refresh_comparison_summary():
    """Existing scenes will be automatically merged after completion; unrun scenes will not generate false placeholder values.."""
    rows = []
    for path in sorted(OUT_ROOT.glob('*/summary.csv')):
        with open(path, 'r', encoding='utf-8') as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        return
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _saved_fullnet_time(out):
    path = out / 'fullnet' / 'training_time.npz'
    if not path.exists():
        return np.nan
    with np.load(path, allow_pickle=True) as timing:
        for key in ('fullnet_total_time', 'total_train_time', 'total_time'):
            if key in timing.files:
                return float(np.asarray(timing[key]))
    return np.nan


def postprocess_existing_results(margin=CURRENT_MARGIN):
    """No retraining/Solve, from the formal primitiveNPZRebuild the summary table and redraw the picture."""
    rows = {}
    for scenario in ('baseline', 'thermal'):
        out = scenario_dir(scenario, margin)
        data_path = out / 'evaluation_data.npz'
        if not data_path.exists():
            raise FileNotFoundError(
                f'Lack of formal evaluation data: {data_path}\n'
                'Please complete the corresponding scene training first. You cannot use calibration orpreviewdata substitution.')
        with np.load(data_path, allow_pickle=True) as evaluation:
            row = evaluation_summary_row(
                evaluation, scenario, margin,
                total_time=_saved_fullnet_time(out))
        save_summary(row, out)
        rows[scenario] = row
        print(
            f"[postprocess] {row['scenario']}: "
            f"coverage={row['coverage_mean']:.3%}, "
            f"feas={row['feas_mean']:.3e}, "
            f"opt={row['opt_mean']:.3e}")

    refresh_comparison_summary()
    baseline = rows['baseline']
    thermal = rows['thermal']
    area_reduction = (
        100.0 * (baseline['nominal_true_region_area']
                 - thermal['nominal_true_region_area'])
        / baseline['nominal_true_region_area'])
    evidence = {
        'case': CASENAME,
        'current_limit_definition':
            f'Imax={margin:.2f}*I_nominal_PYPOWER',
        'current_margin': margin,
        'test_conditions': N_TEST_DTHETA,
        'directions_per_condition': N_DIRS_EVAL,
        'baseline_nominal_true_area': baseline['nominal_true_region_area'],
        'thermal_nominal_true_area': thermal['nominal_true_region_area'],
        'nominal_area_reduction_percent': area_reduction,
        'baseline_feas_mean': baseline['feas_mean'],
        'thermal_feas_mean': thermal['feas_mean'],
        'baseline_opt_mean': baseline['opt_mean'],
        'thermal_opt_mean': thermal['opt_mean'],
        'baseline_coverage_mean': baseline['coverage_mean'],
        'thermal_coverage_mean': thermal['coverage_mean'],
        'thermal_any_ge99_direction_rate':
            thermal['any_ge99_direction_rate'],
        'thermal_ge99_line_ratio_mean': thermal['ge99_line_ratio_mean'],
        'thermal_ge99_line_ratio_median': thermal['ge99_line_ratio_median'],
        'thermal_ge99_line_ratio_p95': thermal['ge99_line_ratio_p95'],
        'thermal_true_support_success_rate':
            thermal['true_support_success_rate'],
        'thermal_feasibility_success_rate':
            thermal['feasibility_success_rate'],
        'thermal_optimality_success_rate':
            thermal['optimality_success_rate'],
        'limit_interpretation':
            'deliberately_tight_congestion_stress_test_not_actual_ampacity',
    }
    evidence_path = OUT_ROOT / 'reviewer_evidence_summary.csv'
    with open(evidence_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(evidence))
        writer.writeheader()
        writer.writerow(evidence)

    from Simulator.draw_pictures.R1_3_thermal_plot import main as plot_main
    plot_main(margin)
    preview_data = (
        OUT_ROOT / f'preview_thermal_{int(round(100*margin)):03d}pct'
        / 'true_region_preview_data.npz')
    if preview_data.exists():
        from Simulator.draw_pictures.R1_3_thermal_region_preview_plot import \
            plot_preview
        plot_preview(preview_data)
    print(f'[postprocess] Unified summary: {OUT_ROOT / "summary.csv"}')
    print(f'[postprocess] Review evidence form: {evidence_path}')
    return evidence_path


def main():
    parser = argparse.ArgumentParser(description='R1.3 case33Branch Circuit Thermal Constraint Stress Test')
    parser.add_argument('--scenario', choices=('both', 'baseline', 'thermal'),
                        default=None,
                        help='Training phase scenario; current defaultbothand write independentv2Directory')
    parser.add_argument('--stage',
                        choices=('preview', 'calibrate', 'all', 'pretrain',
                                 'fullnet', 'postprocess'),
                        default=None,
                        help=('Current defaultall: Pre-training, complete training and unified evaluation; '
                              'postprocessOnly rebuild existing official summaries and images'))
    parser.add_argument('--margin', type=float, default=None,
                        help='thermalscene Imax/I_nominal, Default1.20')
    parser.add_argument(
        '--margins', type=parse_margin_list,
        default=CALIBRATION_MARGINS,
        help='calibrateCandidate magnifications, comma separated; default1.10,1.15,1.20,1.25,1.30,1.40,1.50')
    parser.add_argument('--calibration-dtheta', type=int,
                        default=N_CALIBRATION_DTHETA,
                        help='calibrateNumber of operating conditions, default50 (The first one is the nominal operating condition) ')
    parser.add_argument('--calibration-directions', type=int,
                        default=N_CALIBRATION_DIRS,
                        help='Number of true domain directions for each calibration case, default8')
    parser.add_argument('--min-direction-success', type=float,
                        default=MIN_DIRECTION_SUCCESS_RATE,
                        help='The minimum direction solution success rate required for recommended magnification, default0.995')
    parser.add_argument('--preview-directions', type=int,
                        default=N_PREVIEW_DIRS,
                        help='previewNumber of true feasible region directions, default100')
    args = parser.parse_args()
    scenario = args.scenario or SCENARIO
    stage = args.stage or STAGE
    margin = args.margin if args.margin is not None else CURRENT_MARGIN

    if stage == 'postprocess':
        print('=' * 72)
        print(f'R1.3 Existing formal data post-processing calculation examples: {CASENAME}')
        print('No retraining, no resolving, only summary and picture reconstruction')
        print('=' * 72)
        postprocess_existing_results(margin)
        return

    if stage == 'preview':
        print('=' * 72)
        print(f"R1.3 Thermal Constraint Real Domain Preview Calculation Example: {CASENAME}  Equipment: {DEVICE}")
        print(f"heat capacity ratio: {margin:.3f}, Number of directions: {args.preview_directions}")
        print('=' * 72)
        run_true_region_preview(margin, args.preview_directions)
        return

    if stage == 'calibrate':
        print('=' * 72)
        print(f"R1.3 Heat capacity magnification pre-screening calculation example: {CASENAME}  Equipment: {DEVICE}")
        print(f"Candidate magnification: {', '.join(f'{x:.2f}' for x in args.margins)}")
        print('=' * 72)
        run_margin_calibration(
            args.margins,
            n_dtheta=args.calibration_dtheta,
            n_dirs=args.calibration_directions,
            min_success=args.min_direction_success)
        return

    if scenario in ('both', 'thermal') and margin <= 1.0:
        raise ValueError('--margin must be greater than1, For example1.10')

    scenarios = ('baseline', 'thermal') if scenario == 'both' else (scenario,)
    print('=' * 72)
    print(f"R1.3 One-click experiment calculation example: {CASENAME}  Equipment: {DEVICE}")
    print(f"scene: {', '.join(scenarios)}  stage: {stage}  Thermal constraint magnification: {margin:.3f}")
    print('=' * 72)

    overall_start = time.time()
    for index, current_scenario in enumerate(scenarios, 1):
        tag = scenario_tag(current_scenario, margin)
        print(f"\n{'=' * 28} scene {index}/{len(scenarios)}: {tag} {'=' * 28}")
        if current_scenario == 'thermal':
            print(
                f"thermal constraints: Imax={margin:.3f}*I_nominal(PYPOWER), "
                f"model constraints I2<=Imax^2")
        else:
            print('Thermal Constraints: Off (Original Baseline) ')

        if stage in ('all', 'pretrain'):
            run_pretrain(current_scenario, margin)
        if stage in ('all', 'fullnet'):
            run_fullnet(current_scenario, margin)

    # Generate the comparison figure when both baseline and thermal-limit evaluations are available.
    if stage == 'all' and scenario in ('both', 'thermal'):
        baseline_eval = scenario_dir('baseline', margin) / 'evaluation_data.npz'
        thermal_eval = scenario_dir('thermal', margin) / 'evaluation_data.npz'
        if baseline_eval.exists() and thermal_eval.exists():
            print('\n[plot] baselinewiththermalThe evaluation data has been completed and the formal comparison chart has been generated....')
            from Simulator.draw_pictures.R1_3_thermal_plot import main as plot_main
            plot_main(margin)
        else:
            print(
                '\n[plot] No formal comparison chart is generated yet: missingbaselineorthermalEvaluation data.\n'
                f'  baseline: {baseline_eval}\n'
                f'  thermal:  {thermal_eval}')

    elapsed = time.time() - overall_start
    print('\n' + '=' * 72)
    print(f"R1.3 All experiments are completed, total time taken {elapsed:.1f}s ({elapsed / 3600.0:.2f}h) ")
    print(f"All results: {OUT_ROOT}")
    print('=' * 72)


if __name__ == '__main__':
    main()
