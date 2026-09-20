# -*- coding: utf-8 -*-
"""Compare radial and directional-support boundaries for the case36 three-phase system."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from Simulator import PROJECT_ROOT
from Simulator.cases import DS_case_3phase
from Simulator.reference_point import ReferencePointCalculator


CASENAME = 'case36real_3phase_ds'
N_DIRECTIONS = 360
SCAN_STEP = 1e-3
SCAN_MAX_STEPS = 10000
BISECTION_TOL = 1e-6
BISECTION_MAX_ITER = 60
FEASIBILITY_DISTANCE_TOL = 1e-5

OUT_DIR = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'supervised' / CASENAME / 'radial_vs_support_boundary'
)


def update_parameters(error_calculator, init_params_dict, dtheta):
    """flattendthetaAccording to three phasesPd/QThe original shape is written into the real physical model."""
    pd0 = init_params_dict['Pd_meta']['initial_value']
    qd0 = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd0.size
    pd = pd0 + np.asarray(dtheta[:n_pd]).reshape(pd0.shape)
    qd = qd0 + np.asarray(dtheta[n_pd:]).reshape(qd0.shape)
    error_calculator.update_parameters({'Pd_meta': pd, 'Qd_meta': qd})


def point_is_feasible(error_calculator, point, distance_tol):
    """Determine whether it is feasible based on the projection distance from the target point to the real operating domain."""
    projection = error_calculator.project(np.asarray(point, dtype=float))
    if projection is None or not np.all(np.isfinite(projection)):
        return False
    return float(np.linalg.norm(np.asarray(point) - projection)) <= distance_tol


def find_first_boundary(error_calculator, reference, direction, scan_step,
                        scan_max_steps, bisection_tol, distance_tol):
    """Scan outwards with a fixed step size and find the first feasible/Two points within the infeasible interval."""
    reference = np.asarray(reference, dtype=float).reshape(2)
    direction = np.asarray(direction, dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-15:
        return None, None, 0
    direction = direction / norm

    k_feasible = 0.0
    k_infeasible = None
    calls = 0
    for step_index in range(1, scan_max_steps + 1):
        k_trial = step_index * scan_step
        feasible = point_is_feasible(
            error_calculator, reference + k_trial * direction, distance_tol)
        calls += 1
        if not feasible:
            k_infeasible = k_trial
            break
        k_feasible = k_trial
    if k_infeasible is None:
        return None, None, calls

    for _ in range(BISECTION_MAX_ITER):
        if k_infeasible - k_feasible <= bisection_tol:
            break
        k_mid = 0.5 * (k_feasible + k_infeasible)
        feasible = point_is_feasible(
            error_calculator, reference + k_mid * direction, distance_tol)
        calls += 1
        if feasible:
            k_feasible = k_mid
        else:
            k_infeasible = k_mid
    return reference + k_feasible * direction, k_feasible, calls


def load_case(device):
    """buildcase36Three-phase physical model."""
    ppc = DS_case_3phase.case36real_3phase_ds()
    case = DS_case_3phase.DScase_3phase_train(
        casedata=ppc, model_type='fullnet', plot_flag=False, device=device)
    return ppc, case


def plot_comparison(radial_points_mva, support_points_mva, reference_mva,
                    delta, output_png, output_pdf):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    radial_closed = np.vstack((radial_points_mva, radial_points_mva[0]))
    support_closed = np.vstack((support_points_mva, support_points_mva[0]))

    ax.fill(radial_closed[:, 0], radial_closed[:, 1],
            color='tab:red', alpha=0.14)
    ax.plot(radial_closed[:, 0], radial_closed[:, 1],
            color='tab:red', lw=1.8,
            label='First radial boundary (scan + bisection)')
    ax.fill(support_closed[:, 0], support_closed[:, 1],
            color='tab:blue', alpha=0.10)
    ax.plot(support_closed[:, 0], support_closed[:, 1],
            color='tab:blue', lw=1.5,
            label=r'Support-point boundary ($\max\ d^T x$)')
    ax.scatter(support_points_mva[:, 0], support_points_mva[:, 1],
               color='tab:blue', s=5, alpha=0.55)
    ax.plot(reference_mva[0], reference_mva[1], 'k*', markersize=11,
            label='No-flexibility reference point')

    ax.set_xlabel(r'$P_{\mathrm{TDI}}$ (MW)')
    ax.set_ylabel(r'$Q_{\mathrm{TDI}}$ (MVAr)')
    ax.set_title(f'case36 three-phase, uniform $\\Delta\\theta={delta:+.2f}$')
    ax.grid(True, alpha=0.25)
    ax.axis('equal')
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='case36Comparison diagram of three-phase first radial boundary and support point boundary')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='apply to alldthetaThe uniform change amount of the channel, default0')
    parser.add_argument('--n-directions', type=int, default=N_DIRECTIONS)
    parser.add_argument('--scan-step', type=float, default=SCAN_STEP)
    parser.add_argument('--scan-max-steps', type=int, default=SCAN_MAX_STEPS)
    parser.add_argument('--bisection-tol', type=float, default=BISECTION_TOL)
    parser.add_argument('--feasibility-distance-tol', type=float,
                        default=FEASIBILITY_DISTANCE_TOL)
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if args.n_directions < 3:
        raise ValueError('--n-directionsMust be at least3.')
    if args.scan_step <= 0 or args.bisection_tol <= 0:
        raise ValueError('Scan step size and bisection tolerance must be positive numbers.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Load{CASENAME}, device={device}')
    print(f'Number of search directions in the same group={args.n_directions}, Radial step size={args.scan_step:g} p.u.')
    ppc, case = load_case(device)
    ec = case['errorcalculator'].copy()
    ec.solver.options['tol'] = 1e-9
    ec.solver.options['max_iter'] = 10000
    init_params = case['params']['params_dict']
    dtheta = np.full(case['params']['count'], args.delta, dtype=np.float64)
    update_parameters(ec, init_params, dtheta)

    reference_calculator = ReferencePointCalculator(
        ppc=ppc,
        init_params_dict=init_params,
        original_model=ec.original_model,
        solver='ipopt',
    )
    reference = reference_calculator.compute(dtheta)
    if reference is None or not np.all(np.isfinite(reference)):
        raise RuntimeError('case36Three-phase no-flexibility reference point solution failed.')
    reference = np.asarray(reference, dtype=float)

    angles = np.linspace(0.0, 2.0*np.pi, args.n_directions, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    radial_points = np.full((args.n_directions, 2), np.nan)
    radial_lengths = np.full(args.n_directions, np.nan)
    radial_solver_calls = np.zeros(args.n_directions, dtype=int)

    t0 = time.perf_counter()
    for i, direction in enumerate(directions):
        point, length, calls = find_first_boundary(
            ec, reference, direction,
            scan_step=args.scan_step,
            scan_max_steps=args.scan_max_steps,
            bisection_tol=args.bisection_tol,
            distance_tol=args.feasibility_distance_tol,
        )
        if point is None:
            raise RuntimeError(
                f'direction{i} ({np.degrees(angles[i]):.1f}°) First boundary not found; '
                'Please increase--scan-max-stepsor check the solver.')
        radial_points[i] = point
        radial_lengths[i] = length
        radial_solver_calls[i] = calls
        print(f'radial boundary {i+1:03d}/{args.n_directions}: '
              f'angle={np.degrees(angles[i]):6.1f}°, k={length:.6f}, '
              f'calls={calls}')
    radial_time = time.perf_counter() - t0

    # optimize_direction(c)inErrorCalculatorZhongqiuargmin c^T x.
    # Therefore, for the outward unit directiondincoming-d, getargmax d^T xsupport point.
    support_points = np.full((args.n_directions, 2), np.nan)
    support_objective_directions = -directions
    t1 = time.perf_counter()
    for i, direction in enumerate(directions):
        point = ec.optimize_direction(-direction, in_approx=False)
        if point is None or not np.all(np.isfinite(point)):
            raise RuntimeError(
                f'direction{i} ({np.degrees(angles[i]):.1f}°) The support point solution failed.')
        support_points[i] = np.asarray(point, dtype=float)
        print(f'support point   {i+1:03d}/{args.n_directions}: '
              f'angle={np.degrees(angles[i]):6.1f}°, '
              f'P={point[0]:.6f}, Q={point[1]:.6f}')
    support_time = time.perf_counter() - t1

    support_offsets = support_points - reference
    support_normal_projections = np.einsum(
        'ij,ij->i', support_offsets, directions)
    tangent_directions = np.column_stack((-directions[:, 1], directions[:, 0]))
    support_tangential_offsets = np.einsum(
        'ij,ij->i', support_offsets, tangent_directions)
    pointwise_distances = np.linalg.norm(support_points - radial_points, axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'case36_radial_vs_support_360_delta_{args.delta:+.2f}'
    if args.n_directions != 360:
        stem = f'case36_radial_vs_support_{args.n_directions}_delta_{args.delta:+.2f}'
    data_path = args.output_dir / f'{stem}.npz'
    png_path = args.output_dir / f'{stem}.png'
    pdf_path = args.output_dir / f'{stem}.pdf'
    config_path = args.output_dir / f'{stem}.json'
    base_mva = float(ppc['baseMVA'])

    np.savez(
        data_path,
        dtheta=dtheta,
        reference_point=reference,
        directions=directions,
        angles=angles,
        radial_lengths=radial_lengths,
        radial_boundary_points=radial_points,
        support_boundary_points=support_points,
        support_objective_directions=support_objective_directions,
        support_normal_projections=support_normal_projections,
        support_tangential_offsets=support_tangential_offsets,
        radial_to_support_point_distances=pointwise_distances,
        baseMVA=np.array(base_mva),
        radial_boundary_points_MVA=radial_points*base_mva,
        support_boundary_points_MVA=support_points*base_mva,
        reference_point_MVA=reference*base_mva,
        radial_solver_calls=radial_solver_calls,
        radial_boundary_time=np.array(radial_time),
        support_boundary_time=np.array(support_time),
    )
    plot_comparison(
        radial_points*base_mva, support_points*base_mva, reference*base_mva,
        args.delta, png_path, pdf_path)
    with config_path.open('w', encoding='utf-8') as stream:
        json.dump({
            'case': CASENAME,
            'delta': args.delta,
            'n_directions': args.n_directions,
            'radial_boundary_definition': (
                'unit direction, fixed-step outward feasibility scan, first '
                'infeasible bracket, feasible-side bisection boundary'),
            'support_boundary_definition': (
                'for each outward unit direction d, solve max d^T x by calling '
                'ErrorCalculator.optimize_direction(-d)'),
            'comparison_note': (
                'radial points lie on rays from the reference point; support '
                'points generally do not lie on those rays'),
            'scan_step_pu': args.scan_step,
            'scan_max_steps': args.scan_max_steps,
            'bisection_tolerance_pu': args.bisection_tol,
            'feasibility_distance_tolerance_pu': args.feasibility_distance_tol,
            'device': str(device),
            'mean_radial_to_support_distance_pu': float(np.mean(pointwise_distances)),
            'max_radial_to_support_distance_pu': float(np.max(pointwise_distances)),
        }, stream, ensure_ascii=False, indent=2)

    print(f'\nRadial first boundary calculation time: {radial_time:.2f}s')
    print(f'Support point boundary calculation time: {support_time:.2f}s')
    print(f'Average distance between two types of points indexed in the same direction: {np.mean(pointwise_distances):.6f} p.u.')
    print(f'Maximum distance between two types of points indexed in the same direction: {np.max(pointwise_distances):.6f} p.u.')
    print(f'data: {data_path}')
    print(f'pictures: {png_path}')
    print(f'PDF: {pdf_path}')


if __name__ == '__main__':
    main()
