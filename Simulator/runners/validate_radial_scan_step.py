# -*- coding: utf-8 -*-
"""Validate the radial boundary-scan step sizes used for case33 and case36."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import numpy as np
import torch

from Simulator import PROJECT_ROOT
from Simulator.cases import DS_case_3phase, TD_case
from Simulator.reference_point import ReferencePointCalculator


FINE_STEP = 1e-3
CASE33_CANDIDATE_STEP = 3e-2
CASE36_CANDIDATE_STEP = 2e-3
N_DIRECTIONS = 36
BISECTION_TOL = 1e-6
BISECTION_MAX_ITER = 60
FEASIBILITY_DISTANCE_TOL = 1e-5
AGREEMENT_TOL = 1e-4
DTHETA_RANGE = (-0.5, 0.5)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'supervised' / 'radial_scan_step_validation'
)


def update_parameters(error_calculator, init_params_dict, dtheta):
    """flatten dtheta press Pd/Q The original shape is written into the physical model."""
    pd0 = init_params_dict['Pd_meta']['initial_value']
    qd0 = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd0.size
    pd = pd0 + np.asarray(dtheta[:n_pd]).reshape(pd0.shape)
    qd = qd0 + np.asarray(dtheta[n_pd:]).reshape(qd0.shape)
    error_calculator.update_parameters({'Pd_meta': pd, 'Qd_meta': qd})


def point_is_feasible(error_calculator, point, distance_tol):
    """Determine feasibility based on the projection distance from the target point to the real feasible region."""
    point = np.asarray(point, dtype=float)
    projection = error_calculator.project(point)
    if projection is None or not np.all(np.isfinite(projection)):
        return False
    return float(np.linalg.norm(point - np.asarray(projection))) <= distance_tol


def find_first_boundary(error_calculator, reference, direction, scan_step,
                        max_radius, bisection_tol, distance_tol):
    """Sweep outwards with a fixed step size in the unit direction, and then divide into two parts in the first interval with different signs.."""
    reference = np.asarray(reference, dtype=float).reshape(2)
    direction = np.asarray(direction, dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-15:
        return None, None, 0
    direction = direction / norm

    max_steps = int(np.ceil(max_radius / scan_step))
    k_feasible = 0.0
    k_infeasible = None
    calls = 0
    for scan_index in range(1, max_steps + 1):
        k_trial = scan_index * scan_step
        calls += 1
        if not point_is_feasible(
                error_calculator, reference + k_trial * direction,
                distance_tol):
            k_infeasible = k_trial
            break
        k_feasible = k_trial
    if k_infeasible is None:
        return None, None, calls

    for _ in range(BISECTION_MAX_ITER):
        if k_infeasible - k_feasible <= bisection_tol:
            break
        k_mid = 0.5 * (k_feasible + k_infeasible)
        calls += 1
        if point_is_feasible(
                error_calculator, reference + k_mid * direction,
                distance_tol):
            k_feasible = k_mid
        else:
            k_infeasible = k_mid
    return reference + k_feasible * direction, k_feasible, calls


def build_case(case_name, device):
    """Build consistent with the training script case33 single phase or case36 Three-phase physical model."""
    if case_name == 'case33':
        ppc = TD_case.case33bw_ds()
        case = TD_case.DScase_train(
            casedata=ppc, model_type='fullnet', plot_flag=False,
            device=device)
        full_name = 'case33bw_ds'
    elif case_name == 'case36':
        ppc = DS_case_3phase.case36real_3phase_ds()
        case = DS_case_3phase.DScase_3phase_train(
            casedata=ppc, model_type='fullnet', plot_flag=False,
            device=device)
        full_name = 'case36real_3phase_ds'
    else:
        raise ValueError(f'unknown case: {case_name}')
    return full_name, ppc, case


def scenario_candidates(dim_theta, n_random, seed, dtheta_low, dtheta_high,
                        max_random_attempts):
    """Return zero perturbation first, then generate random candidates; invalid reference points will be filtered later."""
    yield 0, np.zeros(dim_theta, dtype=float), True
    rng = np.random.RandomState(seed)
    for attempt in range(1, max_random_attempts + 1):
        dtheta = rng.uniform(dtheta_low, dtheta_high, size=dim_theta)
        yield attempt, dtheta, False
        if attempt >= max_random_attempts or n_random <= 0:
            return


def save_rows(rows, output_dir):
    """Every time a operating condition is completed, the intermediate results are overwritten and written to facilitate verification after long task interruptions.."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'direction_comparisons.json'
    csv_path = output_dir / 'direction_comparisons.csv'
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    if rows:
        with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def validate_case(case_name, args, device, rows):
    full_name, ppc, case = build_case(case_name, device)
    candidate_step = (args.case33_step if case_name == 'case33'
                      else args.case36_step)
    init_params = case['params']['params_dict']
    ec = case['errorcalculator'].copy()
    ec.solver.options['tol'] = 1e-9
    ec.solver.options['max_iter'] = 10000
    reference_calculator = ReferencePointCalculator(
        ppc=ppc,
        init_params_dict=init_params,
        original_model=ec.original_model,
        solver='ipopt')

    angles = np.linspace(0.0, 2.0*np.pi, args.n_directions,
                         endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    target_scenarios = 1 + args.n_random
    accepted = 0
    rejected_references = 0
    started = time.perf_counter()

    candidates = scenario_candidates(
        case['params']['count'], args.n_random, args.seed,
        args.dtheta_low, args.dtheta_high, args.max_random_attempts)
    for attempt, dtheta, is_zero in candidates:
        if accepted >= target_scenarios:
            break
        update_parameters(ec, init_params, dtheta)
        reference = reference_calculator.compute(dtheta)
        if reference is None or not np.all(np.isfinite(reference)):
            rejected_references += 1
            if is_zero:
                raise RuntimeError(f'{full_name} The zero perturbation reference point solution failed.')
            print(f'[{full_name}] random candidate{attempt}Reference point failed, skipped.')
            continue
        reference = np.asarray(reference, dtype=float).reshape(2)
        scenario_index = accepted
        accepted += 1
        print(f'[{full_name}] Working conditions {accepted}/{target_scenarios}, '
              f'candidate={attempt}, Start comparing{args.n_directions}direction; '
              f'base step size={args.fine_step:g}, candidate step size={candidate_step:g}.')

        for direction_index, (angle, direction) in enumerate(
                zip(angles, directions)):
            fine_point, fine_length, fine_calls = find_first_boundary(
                ec, reference, direction, args.fine_step, args.max_radius,
                args.bisection_tol, args.feasibility_distance_tol)
            candidate_point, candidate_length, candidate_calls = find_first_boundary(
                ec, reference, direction, candidate_step, args.max_radius,
                args.bisection_tol, args.feasibility_distance_tol)

            success = (fine_point is not None and candidate_point is not None)
            if success:
                radial_error = abs(float(candidate_length - fine_length))
                coordinate_error = float(np.linalg.norm(
                    np.asarray(candidate_point) - np.asarray(fine_point)))
                passed = (radial_error <= args.agreement_tol
                          and coordinate_error <= args.agreement_tol)
            else:
                radial_error = coordinate_error = float('nan')
                passed = False

            rows.append({
                'case': full_name,
                'scenario_index': scenario_index,
                'candidate_attempt': attempt,
                'is_zero_dtheta': bool(is_zero),
                'direction_index': direction_index,
                'angle_deg': float(np.degrees(angle)),
                'fine_step_pu': args.fine_step,
                'candidate_step_pu': candidate_step,
                'reference_p': float(reference[0]),
                'reference_q': float(reference[1]),
                'fine_length_pu': (None if fine_length is None
                                   else float(fine_length)),
                'candidate_length_pu': (None if candidate_length is None
                                        else float(candidate_length)),
                'radial_error_pu': (None if not np.isfinite(radial_error)
                                    else radial_error),
                'coordinate_error_pu': (
                    None if not np.isfinite(coordinate_error)
                    else coordinate_error),
                'fine_solver_calls': int(fine_calls),
                'candidate_solver_calls': int(candidate_calls),
                'passed': bool(passed),
            })
            print(f'  {direction_index + 1:02d}/{args.n_directions}: '
                  f'fine={fine_length}, candidate={candidate_length}, '
                  f'error={radial_error:.3e}, passed={passed}')

        save_rows(rows, args.output_dir)

    if accepted < target_scenarios:
        raise RuntimeError(
            f'{full_name}only get{accepted}/{target_scenarios}valid operating conditions; '
            f'Please increase--max-random-attemptsor zoom outdthetarange.')
    return {
        'case': full_name,
        'tested_scenarios': accepted,
        'tested_directions': accepted * args.n_directions,
        'candidate_step_pu': candidate_step,
        'rejected_reference_candidates': rejected_references,
        'elapsed_seconds': time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(
        description=('to0.001as a benchmark to verifycase33=0.03, '
                     'case36=0.002The radial scan step size of'))
    parser.add_argument('--cases', nargs='+', choices=['case33', 'case36'],
                        default=['case33', 'case36'])
    parser.add_argument('--n-random', type=int, default=2,
                        help='eachcaseNumber of effective random operating conditions other than zero disturbance')
    parser.add_argument('--n-directions', type=int, default=N_DIRECTIONS)
    parser.add_argument('--fine-step', type=float, default=FINE_STEP)
    parser.add_argument('--case33-step', type=float,
                        default=CASE33_CANDIDATE_STEP,
                        help='case33Candidate radial scan step size, default0.03 p.u.')
    parser.add_argument('--case36-step', type=float,
                        default=CASE36_CANDIDATE_STEP,
                        help='case36Candidate radial scan step size, default0.002 p.u.')
    parser.add_argument('--agreement-tol', type=float, default=AGREEMENT_TOL,
                        help='Consistency threshold for two step boundaries (p.u.) ')
    parser.add_argument('--bisection-tol', type=float, default=BISECTION_TOL)
    parser.add_argument('--feasibility-distance-tol', type=float,
                        default=FEASIBILITY_DISTANCE_TOL)
    parser.add_argument('--max-radius', type=float, default=10.0)
    parser.add_argument('--dtheta-low', type=float, default=DTHETA_RANGE[0])
    parser.add_argument('--dtheta-high', type=float, default=DTHETA_RANGE[1])
    parser.add_argument('--max-random-attempts', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.n_random < 0 or args.n_directions < 3:
        raise ValueError('n-randomMust be non-negative, n-directionsMust be at least3.')
    positive = [args.fine_step, args.case33_step, args.case36_step,
                args.agreement_tol,
                args.bisection_tol, args.feasibility_distance_tol,
                args.max_radius]
    if any(value <= 0 for value in positive):
        raise ValueError('step size, tolerance andmax-radiusMust be a positive number.')
    for case_label, candidate_step in (
            ('case33-step', args.case33_step),
            ('case36-step', args.case36_step)):
        ratio = candidate_step / args.fine_step
        if abs(ratio - round(ratio)) > 1e-12:
            raise ValueError(
                f'{case_label}must befine-stepAn integer multiple of , to facilitate strict comparison.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}; base step size={args.fine_step}; '
          f'case33candidate step size={args.case33_step}; '
          f'case36candidate step size={args.case36_step}; '
          f'consistency threshold={args.agreement_tol}')
    rows = []
    case_metadata = []
    for case_name in args.cases:
        case_metadata.append(validate_case(case_name, args, device, rows))

    summaries = []
    for metadata in case_metadata:
        case_rows = [row for row in rows if row['case'] == metadata['case']]
        errors = [row['radial_error_pu'] for row in case_rows
                  if row['radial_error_pu'] is not None]
        all_passed = (len(case_rows) == metadata['tested_directions']
                      and all(row['passed'] for row in case_rows))
        fine_calls = sum(row['fine_solver_calls'] for row in case_rows)
        candidate_calls = sum(
            row['candidate_solver_calls'] for row in case_rows)
        summary = dict(metadata)
        summary.update({
            'fine_step_pu': args.fine_step,
            'candidate_step_pu': metadata['candidate_step_pu'],
            'agreement_tolerance_pu': args.agreement_tol,
            'max_radial_error_pu': max(errors) if errors else None,
            'mean_radial_error_pu': float(np.mean(errors)) if errors else None,
            'fine_solver_calls': fine_calls,
            'candidate_solver_calls': candidate_calls,
            'solver_call_reduction_fraction': (
                1.0 - candidate_calls / fine_calls if fine_calls else None),
            'verdict': ('SAFE_ON_TESTED_SCENARIOS' if all_passed
                        else 'NOT_SAFE_ON_TESTED_SCENARIOS'),
        })
        summaries.append(summary)

    report = {
        'interpretation': (
            'Empirical sampled validation, not a proof over every dtheta. '
            'SAFE means every tested direction agreed with the 0.001-p.u. '
            'baseline within agreement_tolerance_pu.'),
        'settings': vars(args) | {'output_dir': str(args.output_dir)},
        'cases': summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / 'summary.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    save_rows(rows, args.output_dir)

    print('\nfinal judgment: ')
    for summary in summaries:
        print(f"  {summary['case']}: {summary['verdict']}, "
              f"max_error={summary['max_radial_error_pu']} p.u., "
              f"Solve calls reduced={summary['solver_call_reduction_fraction']:.1%}")
    print(f'Summary: {report_path}')


if __name__ == '__main__':
    main()
