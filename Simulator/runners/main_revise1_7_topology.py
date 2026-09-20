# -*- coding: utf-8 -*-
"""Run the R1.7 zero-shot topology-change evaluation for case33."""
import argparse
import copy
import csv
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Compatible ``python Simulator/runners/main_revise1_7_topology.py`` with ``python -m``.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Simulator import PROJECT_ROOT
from Simulator.Approximator import PreTrainNet, FullNet
from Simulator.cases import TD_case
from Simulator.reference_point import (
    REFERENCE_MEMBERSHIP_TOL,
    reference_point_membership,
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CASENAME = 'case33bw_ds'

WEIGHTS_DIR = (
    PROJECT_ROOT / 'results' / 'ds_proj_paper' / CASENAME /
    'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)')
OUT = (PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' /
       'comparison' / 'topology' / CASENAME)

N_DTHETA = 20
N_DIRECTIONS = 36
DTHETA_RANGE = (-0.5, 0.5)
SEED = 42

# Each topology closes one tie line and opens one branch in the resulting loop.
TOPOLOGY_SCHEMES = {
    'Base': {'close': None, 'open': None},
    'T1': {'close': (21, 8), 'open': (7, 8)},
    'T2': {'close': (9, 15), 'open': (14, 15)},
    'T3': {'close': (12, 22), 'open': (21, 22)},
    'T4': {'close': (18, 33), 'open': (17, 18)},
    'T5': {'close': (25, 29), 'open': (28, 29)},
}


def edge_key(edge):
    if edge is None:
        return None
    return frozenset((int(edge[0]), int(edge[1])))


def topology_case(name):
    """Construct the specified switch scheme and start from the root node1Redirect all legs toparent→child."""
    if name not in TOPOLOGY_SCHEMES:
        raise ValueError(f'Unknown topology{name!r}')
    scheme = TOPOLOGY_SCHEMES[name]
    ppc = TD_case.case33bw_ds(radial=False)
    all_branches = np.asarray(ppc['branch'], dtype=float)
    close_key = edge_key(scheme['close'])
    open_key = edge_key(scheme['open'])

    selected = []
    close_found = close_key is None
    open_found = open_key is None
    for original_index, row in enumerate(all_branches):
        key = edge_key((row[0], row[1]))
        active = bool(int(row[10]) == 1)
        if key == close_key:
            if active:
                raise ValueError(f'{name}Ties to be closed{scheme["close"]}Originally closed')
            active = True
            close_found = True
        if key == open_key:
            if not active:
                raise ValueError(f'{name}Branch to be disconnected{scheme["open"]}Originally unclosed')
            active = False
            open_found = True
        if active:
            selected.append((original_index, row.copy()))
    if not close_found or not open_found:
        raise ValueError(f'{name}The branch circuit in the switching scheme is not incase33found in')

    bus_ids = [int(row[0]) for row in ppc['bus']]
    adjacency = {bus: [] for bus in bus_ids}
    for pos, (_, row) in enumerate(selected):
        u, v = int(row[0]), int(row[1])
        adjacency[u].append((v, pos))
        adjacency[v].append((u, pos))

    # BFS verifies connectivity and orients every branch away from the root.
    visited = {1}
    queue = deque([1])
    oriented = []
    used_edges = set()
    while queue:
        parent = queue.popleft()
        for child, pos in adjacency[parent]:
            if pos in used_edges:
                continue
            used_edges.add(pos)
            if child in visited:
                raise ValueError(f'{name}Not radial: Loop edge detected{parent}-{child}')
            visited.add(child)
            queue.append(child)
            original_index, row = selected[pos]
            row[0], row[1] = parent, child
            row[10] = 1.0
            oriented.append((original_index, row))

    n_bus = len(bus_ids)
    if len(visited) != n_bus:
        missing = sorted(set(bus_ids) - visited)
        raise ValueError(f'{name}Disconnected, isolated nodes: {missing}')
    if len(selected) != n_bus - 1 or len(oriented) != n_bus - 1:
        raise ValueError(
            f'{name}not a tree: bus={n_bus}, selected={len(selected)}, '
            f'oriented={len(oriented)}')

    ppc = copy.deepcopy(ppc)
    ppc['branch'] = np.asarray([row for _, row in oriented], dtype=float)
    ppc['topology_name'] = name
    ppc['topology_closed_tie'] = scheme['close']
    ppc['topology_opened_line'] = scheme['open']
    ppc['topology_original_branch_indices'] = np.asarray(
        [idx for idx, _ in oriented], dtype=int)
    return ppc


def power_flow_diagnostic(ppc):
    """The nominal power flow is only used for topology and voltage diagnosis and does not participate in error calculation.."""
    from pypower.api import ppoption, runpf
    pf_case = copy.deepcopy(ppc)
    pf_case['bus'] = np.asarray(pf_case['bus'], dtype=float)
    pf_case['branch'] = np.asarray(pf_case['branch'], dtype=float)
    pf_case['gen'] = np.asarray(pf_case['gen'], dtype=float)
    result, success = runpf(pf_case, ppoption(VERBOSE=0, OUT_ALL=0))
    if not success:
        return False, np.nan, np.nan
    voltage = np.asarray(result['bus'])[:, 7]
    return True, float(np.min(voltage)), float(np.max(voltage))


def load_original_model(base_case):
    required = ['pretrainnet_weights.pth', 'fullnet_weights_feasible.pth']
    missing = [name for name in required if not (WEIGHTS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f'Original weight is missing{missing}: {WEIGHTS_DIR}')
    pre = PreTrainNet(
        base_case['A_hat'], base_case['b_hat'],
        is_epigraph=False, device=DEVICE)
    pre.load_state_dict(torch.load(
        WEIGHTS_DIR / 'pretrainnet_weights.pth', map_location=DEVICE))
    pre = pre.to(DEVICE)
    with torch.no_grad():
        A_pre, b_pre = pre()
    A_pre = A_pre[0].detach().cpu().numpy()
    b_pre = b_pre[0].detach().cpu().numpy()
    model = FullNet(
        dim_theta=base_case['params']['count'],
        A_init=A_pre, b_init=b_pre, n_hidden=128, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(
        WEIGHTS_DIR / 'fullnet_weights_feasible.pth', map_location=DEVICE))
    model.eval()
    return model


def directions(n_directions):
    angle = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)
    return np.column_stack((np.cos(angle), np.sin(angle)))


def test_dthetas(dim_theta, n_dtheta):
    rng = np.random.RandomState(SEED)
    values = rng.uniform(
        DTHETA_RANGE[0], DTHETA_RANGE[1], size=(n_dtheta, dim_theta))
    values[0] = 0.0
    return values


def update_parameters(ec, init_params_dict, dtheta):
    pd_init = init_params_dict['Pd_meta']['initial_value']
    qd_init = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd_init.size
    ec.update_parameters({
        'Pd_meta': pd_init + dtheta[:n_pd].reshape(pd_init.shape),
        'Qd_meta': qd_init + dtheta[n_pd:].reshape(qd_init.shape),
    })


def k_max_polytope(A, b, x_ref, direction):
    av = A @ direction
    slack = b - A @ x_ref
    valid = av > 1e-12
    if not np.any(valid):
        return np.nan
    value = np.min(slack[valid] / av[valid])
    return float(value) if np.isfinite(value) else np.nan


def predict_all(model, dthetas):
    A_values, b_values = [], []
    with torch.inference_mode():
        for dtheta in dthetas:
            tensor = torch.as_tensor(
                dtheta, dtype=torch.float32, device=DEVICE)
            A, b = model(tensor)
            A_values.append(A[0].detach().cpu().numpy())
            b_values.append(b[0].detach().cpu().numpy())
    return np.asarray(A_values), np.asarray(b_values)


def evaluate_topology(name, ppc, predicted_A, predicted_b, dthetas, eval_dirs):
    """Use original topologyPINNOutput evaluation specifies the real physical topology and saves all sequential quantities."""
    print(f'\n[{name}] Build realistic physical models...')
    case = TD_case.DScase_train(
        casedata=ppc, model_type='fullnet', plot_flag=False,
        total_samples=1, batch_size=1, device='cpu')
    ec = case['errorcalculator']
    init_params = case['params']['params_dict']
    ec.attach_radial(ppc, init_params, n_dirs=len(eval_dirs), seed=123)

    shape = (len(dthetas), len(eval_dirs))
    feasibility = np.full(shape, np.nan)
    optimality = np.full(shape, np.nan)
    coverage = np.full(shape, np.nan)
    k_omega = np.full(shape, np.nan)
    k_poly = np.full(shape, np.nan)
    feasibility_success = np.zeros(shape, dtype=bool)
    optimality_success = np.zeros(shape, dtype=bool)
    radial_success = np.zeros(shape, dtype=bool)
    membership_failure = np.zeros(len(dthetas), dtype=bool)
    x_ref_values = np.full((len(dthetas), 2), np.nan)
    true_support = np.full((len(dthetas), len(eval_dirs), 2), np.nan)

    start = time.time()
    for idx, dtheta in enumerate(dthetas):
        update_parameters(ec, init_params, dtheta)
        A = predicted_A[idx]
        b = predicted_b[idx]
        ec.update_polytope(A_hat=A, b_hat=b)

        true_points = []
        for j, direction in enumerate(eval_dirs):
            point = ec.optimize_direction(direction, in_approx=False)
            true_points.append(point)
            if point is not None:
                true_support[idx, j] = point

        for j, direction in enumerate(eval_dirs):
            x_poly = ec.optimize_direction(direction, in_approx=True)
            projected_true = ec.project(x_poly) if x_poly is not None else None
            if x_poly is not None and projected_true is not None:
                feasibility[idx, j] = np.sum((x_poly - projected_true) ** 2)
                feasibility_success[idx, j] = True

            x_true = true_points[j]
            projected_poly = (ec.project(x_true, to_approx=True)
                              if x_true is not None else None)
            if x_true is not None and projected_poly is not None:
                optimality[idx, j] = np.sum((x_true - projected_poly) ** 2)
                optimality_success[idx, j] = True

        radial = ec.calculate_radial(eval_dirs, dtheta)
        if radial is not None:
            x_ref = radial[0]['x_ref']
            x_ref_values[idx] = x_ref
            is_member, _ = reference_point_membership(
                A, b, x_ref, REFERENCE_MEMBERSHIP_TOL)
            membership_failure[idx] = not is_member
            for j, (record, direction) in enumerate(zip(radial, eval_dirs)):
                ko = float(record['k_omega'])
                k_omega[idx, j] = ko
                radial_success[idx, j] = ko > 1e-9
                if is_member and ko > 1e-9:
                    kp = k_max_polytope(A, b, x_ref, direction)
                    k_poly[idx, j] = kp
                    if np.isfinite(kp):
                        coverage[idx, j] = kp / ko

        print(
            f'  Working conditions {idx + 1:02d}/{len(dthetas)}: '
            f'feas={feasibility_success[idx].sum()}/{len(eval_dirs)}, '
            f'opt={optimality_success[idx].sum()}/{len(eval_dirs)}, '
            f'rho={np.isfinite(coverage[idx]).sum()}/{len(eval_dirs)}')

    return {
        'feasibility': feasibility,
        'optimality': optimality,
        'coverage': coverage,
        'k_omega': k_omega,
        'k_poly': k_poly,
        'feasibility_success': feasibility_success,
        'optimality_success': optimality_success,
        'radial_success': radial_success,
        'membership_failure': membership_failure,
        'x_ref': x_ref_values,
        'true_support': true_support,
        'evaluation_time_s': np.asarray(time.time() - start),
    }


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def stats(values):
    values = finite(values)
    if values.size == 0:
        return {
            'mean': np.nan, 'median': np.nan, 'p05': np.nan,
            'p95': np.nan, 'count': 0}
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p05': float(np.percentile(values, 5)),
        'p95': float(np.percentile(values, 95)),
        'count': int(values.size),
    }


SUMMARY_FIELDS = [
    'topology', 'closed_tie', 'opened_line', 'pf_success', 'pf_vmin', 'pf_vmax',
    'coverage_mean', 'coverage_median', 'coverage_p05', 'coverage_p95',
    'undercoverage_mean', 'undercoverage_p95', 'overcoverage_mean',
    'undercovered_direction_rate', 'overcovered_direction_rate',
    'feas_mean', 'feas_median', 'feas_p95',
    'opt_mean', 'opt_median', 'opt_p95',
    'membership_failed_cases', 'reference_membership_success_rate',
    'radial_success_rate', 'feas_success_rate', 'opt_success_rate',
    'evaluation_time_s']


def build_summary(name, result, pf_record):
    rho = finite(result['coverage'])
    feas = stats(result['feasibility'])
    opt = stats(result['optimality'])
    if rho.size:
        under = np.maximum(1.0 - rho, 0.0)
        over = np.maximum(rho - 1.0, 0.0)
        under_rate = float(np.mean(rho < 1.0 - 1e-8))
        over_rate = float(np.mean(rho > 1.0 + 1e-8))
    else:
        under = over = np.array([np.nan])
        under_rate = over_rate = np.nan
    rho_stat = stats(rho)
    under_stat = stats(under)
    scheme = TOPOLOGY_SCHEMES[name]
    total = result['feasibility_success'].size
    n_cases = len(result['membership_failure'])
    return {
        'topology': name,
        'closed_tie': '-' if scheme['close'] is None else f'{scheme["close"][0]}-{scheme["close"][1]}',
        'opened_line': '-' if scheme['open'] is None else f'{scheme["open"][0]}-{scheme["open"][1]}',
        'pf_success': int(pf_record[0]),
        'pf_vmin': pf_record[1],
        'pf_vmax': pf_record[2],
        'coverage_mean': rho_stat['mean'],
        'coverage_median': rho_stat['median'],
        'coverage_p05': rho_stat['p05'],
        'coverage_p95': rho_stat['p95'],
        'undercoverage_mean': under_stat['mean'],
        'undercoverage_p95': under_stat['p95'],
        'overcoverage_mean': float(np.nanmean(over)),
        'undercovered_direction_rate': under_rate,
        'overcovered_direction_rate': over_rate,
        'feas_mean': feas['mean'],
        'feas_median': feas['median'],
        'feas_p95': feas['p95'],
        'opt_mean': opt['mean'],
        'opt_median': opt['median'],
        'opt_p95': opt['p95'],
        'membership_failed_cases': int(np.sum(result['membership_failure'])),
        'reference_membership_success_rate': float(
            1.0 - np.sum(result['membership_failure']) / n_cases),
        'radial_success_rate': float(
            np.sum(result['radial_success']) / result['radial_success'].size),
        'feas_success_rate': float(np.sum(result['feasibility_success']) / total),
        'opt_success_rate': float(np.sum(result['optimality_success']) / total),
        'evaluation_time_s': float(result['evaluation_time_s']),
    }


def save_topology_data(name, ppc, result, dthetas, eval_dirs,
                       predicted_A, predicted_b, quick):
    topology_out = OUT / name
    topology_out.mkdir(parents=True, exist_ok=True)
    scheme = TOPOLOGY_SCHEMES[name]
    np.savez(
        topology_out / 'evaluation_data.npz',
        topology=np.asarray(name),
        closed_tie=np.asarray(scheme['close'] if scheme['close'] else (-1, -1)),
        opened_line=np.asarray(scheme['open'] if scheme['open'] else (-1, -1)),
        branch=ppc['branch'],
        original_branch_indices=ppc['topology_original_branch_indices'],
        dthetas=dthetas,
        eval_dirs=eval_dirs,
        predicted_A=predicted_A,
        predicted_b=predicted_b,
        quick_mode=np.asarray(quick),
        error_definition=np.asarray('squared_euclidean_distance'),
        coverage_definition=np.asarray('reference_point_radial_kP_over_kOmega'),
        **result)


def save_summary(rows):
    with open(OUT / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        OUT / 'summary.npz',
        topology=np.asarray([row['topology'] for row in rows]),
        closed_tie=np.asarray([row['closed_tie'] for row in rows]),
        opened_line=np.asarray([row['opened_line'] for row in rows]),
        **{field: np.asarray([float(row[field]) for row in rows], dtype=float)
           for field in SUMMARY_FIELDS
           if field not in ('topology', 'closed_tie', 'opened_line')})


def postprocess_existing():
    """From formal to formalNPZRebuild summaries and images without retraining or calling the optimization solver."""
    old_rows = {}
    old_summary = OUT / 'summary.csv'
    if old_summary.exists():
        with open(old_summary, 'r', encoding='utf-8') as f:
            old_rows = {row['topology']: row for row in csv.DictReader(f)}

    results = {}
    for name in TOPOLOGY_SCHEMES:
        path = OUT / name / 'evaluation_data.npz'
        if not path.exists():
            raise FileNotFoundError(
                f'No official step-by-step data found: {path}\nPlease run the completeR1.7experiment.')
        with np.load(path, allow_pickle=True) as data:
            if bool(np.asarray(data['quick_mode']).item()):
                raise ValueError(f'{path}Yesquickdata, formal summaries cannot be generated.')
            results[name] = {key: data[key].copy() for key in data.files}

    rows = []
    for name in TOPOLOGY_SCHEMES:
        if name not in old_rows:
            raise FileNotFoundError(
                'oldsummary.csvThere is a lack of power flow diagnostic information in the.')
        old = old_rows[name]
        pf_record = (
            bool(int(float(old['pf_success']))),
            float(old['pf_vmin']), float(old['pf_vmax']))
        rows.append(build_summary(name, results[name], pf_record))
    save_summary(rows)

    from Simulator.draw_pictures.R1_7_topology_plot import main as plot_main
    plot_main(root=OUT)
    print(f'[postprocess] has been removed from the existing20Working conditions×36Orientation data reconstruction: {OUT / "summary.csv"}')


def main():
    global OUT
    parser = argparse.ArgumentParser(description='R1.7 case33Zero-shot topology change testing')
    parser.add_argument('--quick', action='store_true',
                        help='2Working conditions×4Orientation, debugging only, cannot be used for papers')
    parser.add_argument(
        '--postprocess', action='store_true',
        help='Only read existing formalNPZ, Rebuild summaries and graphs without training or invoking the optimization solver')
    args = parser.parse_args()
    if args.quick and args.postprocess:
        parser.error('--quickwith--postprocesscannot be used at the same time.')
    if args.postprocess:
        postprocess_existing()
        return
    if args.quick:
        OUT = OUT / 'quick_smoke'
    OUT.mkdir(parents=True, exist_ok=True)

    n_dtheta = 2 if args.quick else N_DTHETA
    n_directions = 4 if args.quick else N_DIRECTIONS
    eval_dirs = directions(n_directions)

    print('=' * 78)
    print('R1.7 case33Zero-sample topology change experiment')
    print(f'mode: {"QUICK (Cannot be used for papers) " if args.quick else "FORMAL"}')
    print(f'Equipment: {DEVICE}  scale: 6Topology×{n_dtheta}Working conditions×{n_directions}direction')
    print(f'Manuscript weight: {WEIGHTS_DIR}')
    print(f'output: {OUT}')
    print('=' * 78)

    topologies = {}
    pf_records = {}
    for name in TOPOLOGY_SCHEMES:
        ppc = topology_case(name)
        topologies[name] = ppc
        pf_records[name] = power_flow_diagnostic(ppc)
        scheme = TOPOLOGY_SCHEMES[name]
        print(
            f'{name}: close={scheme["close"]}, open={scheme["open"]}, '
            f'branches={len(ppc["branch"])}, PF={pf_records[name][0]}, '
            f'V=[{pf_records[name][1]:.4f},{pf_records[name][2]:.4f}]')

    # Only useBaseThe physical model determines the input dimensions and loadsBasemanuscript network.
    base_case = TD_case.DScase_train(
        casedata=topologies['Base'], model_type='fullnet', plot_flag=False,
        total_samples=1, batch_size=1, device='cpu')
    model = load_original_model(base_case)
    dthetas = test_dthetas(base_case['params']['count'], n_dtheta)
    predicted_A, predicted_b = predict_all(model, dthetas)

    all_results = {}
    overall_start = time.time()
    for name, ppc in topologies.items():
        result = evaluate_topology(
            name, ppc, predicted_A, predicted_b, dthetas, eval_dirs)
        all_results[name] = result
        save_topology_data(
            name, ppc, result, dthetas, eval_dirs,
            predicted_A, predicted_b, args.quick)

    rows = [
        build_summary(name, all_results[name], pf_records[name])
        for name in TOPOLOGY_SCHEMES]
    save_summary(rows)

    print('\n' + '-' * 78)
    for row in rows:
        print(
            f"{row['topology']}: rho={row['coverage_mean']:.4f}, "
            f"under={row['undercoverage_mean']:.4f}, "
            f"over={row['overcoverage_mean']:.4f}, "
            f"feas={row['feas_mean']:.3e}, opt={row['opt_mean']:.3e}, "
            f"radial-success={row['radial_success_rate']:.1%}")

    print('\n[plot] Generate topology change comparison chart...')
    from Simulator.draw_pictures.R1_7_topology_plot import main as plot_main
    plot_main(root=OUT)
    elapsed = time.time() - overall_start
    print('\n' + '=' * 78)
    print(f'All completed, total time taken: {elapsed:.1f}s ({elapsed / 60.0:.1f}min) ')
    print(f'Summary: {OUT / "summary.csv"}')
    print('=' * 78)


if __name__ == '__main__':
    main()
