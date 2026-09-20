# -*- coding: utf-8 -*-
"""Evaluate near-convexity using support points and projected chord samples."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import numpy as np
from scipy.spatial import ConvexHull, QhullError

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Simulator import PROJECT_ROOT
from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase


CASE_NAMES = (
    'case33bw_ds',
    'case118zh_ds',
    'case533mt_hi_ds',
    'case36real_3phase_ds',
)
DEFAULT_CASES = ('case33bw_ds', 'case118zh_ds', 'case533mt_hi_ds')
N_CONDITIONS = 10
N_DIRECTIONS = 360
N_RANDOM_CHORDS = 500
# Dense inspection of adjacent edges of the convex hull to avoid missing narrow infeasible segments near either endpoint.
EDGE_LAMBDAS = tuple(index / 10.0 for index in range(1, 10))
# Evaluate three representative non-adjacent chords to limit the number of physical-model calls.
RANDOM_LAMBDAS = (0.25, 0.50, 0.75)
DTHETA_RANGE = (-0.5, 0.5)
IPOPT_MAX_ITER = 10000
SEED = 42
UNIQUE_DECIMALS = 8

OUT = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'support_chord_convexity'
)


def directions_even(n):
    angles = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    return angles, np.column_stack((np.cos(angles), np.sin(angles)))


def load_case(casename):
    if casename == 'case36real_3phase_ds':
        ppc = DS_case_3phase.case36real_3phase_ds()
        case = DS_case_3phase.DScase_3phase_train(
            casedata=ppc, model_type='fullnet', plot_flag=False,
            total_samples=1, batch_size=1, device='cpu')
    else:
        ppc = getattr(TD_case, casename)()
        case = TD_case.DScase_train(
            casedata=ppc, model_type='fullnet', plot_flag=False,
            total_samples=1, batch_size=1, device='cpu')
    ec = case['errorcalculator']
    if str(ec.solver_str).lower() == 'ipopt':
        ec.solver.options['tol'] = 1e-9
        ec.solver.options['max_iter'] = IPOPT_MAX_ITER
    return ppc, case


def generate_dthetas(dim, n_conditions):
    rng = np.random.RandomState(SEED)
    values = rng.uniform(
        DTHETA_RANGE[0], DTHETA_RANGE[1], size=(n_conditions, dim))
    values[0] = 0.0
    return values


def update_parameters(ec, init_params, dtheta):
    pd0 = init_params['Pd_meta']['initial_value']
    qd0 = init_params['Qd_meta']['initial_value']
    n_pd = pd0.size
    ec.update_parameters({
        'Pd_meta': pd0 + np.asarray(dtheta[:n_pd]).reshape(pd0.shape),
        'Qd_meta': qd0 + np.asarray(dtheta[n_pd:]).reshape(qd0.shape),
    })


def project_distance(ec, target):
    target = np.asarray(target, dtype=float)
    projection = ec.project(target)
    if projection is None or not np.all(np.isfinite(projection)):
        return None, np.nan
    projection = np.asarray(projection, dtype=float)
    return projection, float(np.linalg.norm(target - projection))


def unique_points_with_sources(points, success, decimals=UNIQUE_DECIMALS):
    """Deduplicate successful support points and retain the direction index of the first occurrence of each unique point."""
    valid_indices = np.flatnonzero(success & np.all(np.isfinite(points), axis=1))
    if not len(valid_indices):
        return np.empty((0, 2)), np.empty(0, dtype=int)
    rounded = np.round(points[valid_indices], decimals=decimals)
    _, first = np.unique(rounded, axis=0, return_index=True)
    first = np.sort(first)
    source_indices = valid_indices[first]
    return points[source_indices], source_indices


def nonadjacent_pairs(n_vertices, n_pairs, rng):
    """Extract non-adjacent unordered point pairs from convex hull vertices without replacement."""
    if n_vertices < 4 or n_pairs <= 0:
        return []
    all_pairs = [
        (i, j)
        for i in range(n_vertices)
        for j in range(i + 1, n_vertices)
        if j != i + 1 and not (i == 0 and j == n_vertices - 1)
    ]
    if len(all_pairs) <= n_pairs:
        return all_pairs
    selected = rng.choice(len(all_pairs), size=n_pairs, replace=False)
    return [all_pairs[index] for index in selected]


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        if not fieldnames:
            return
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


LINE_FIELDS = [
    'case', 'condition_index', 'line_type', 'endpoint_index_1',
    'endpoint_index_2', 'lambda', 'endpoint_1_p', 'endpoint_1_q',
    'endpoint_2_p', 'endpoint_2_q', 'target_p', 'target_q',
    'projection_success', 'projection_p', 'projection_q', 'distance_pu',
    'condition_hull_diameter_pu', 'normalized_distance',
]


def evaluate_case(casename, args, output_root):
    print('\n' + '=' * 78)
    print(f'{casename}: {args.n_conditions}Working conditions × '
          f'{args.n_directions}support direction')
    _, case = load_case(casename)
    ec = case['errorcalculator']
    init_params = case['params']['params_dict']
    dthetas = generate_dthetas(case['params']['count'], args.n_conditions)
    angles, directions = directions_even(args.n_directions)
    case_dir = output_root / casename
    case_dir.mkdir(parents=True, exist_ok=True)

    support_points = np.full(
        (args.n_conditions, args.n_directions, 2), np.nan)
    support_success = np.zeros(
        (args.n_conditions, args.n_directions), dtype=bool)
    support_centroids = np.full((args.n_conditions, 2), np.nan)
    hull_diameters = np.full(args.n_conditions, np.nan)
    unique_counts = np.zeros(args.n_conditions, dtype=int)
    hull_vertex_counts = np.zeros(args.n_conditions, dtype=int)
    line_rows = []
    condition_metadata = []
    case_started = time.perf_counter()

    for condition in range(args.n_conditions):
        condition_started = time.perf_counter()
        update_parameters(ec, init_params, dthetas[condition])
        print(f'[{casename}] Working conditions {condition + 1}/{args.n_conditions}: '
              f'Search{args.n_directions}support point')
        for direction_index, direction in enumerate(directions):
            point = ec.optimize_direction(-direction, in_approx=False)
            if point is not None and np.all(np.isfinite(point)):
                support_points[condition, direction_index] = point
                support_success[condition, direction_index] = True
            if ((direction_index + 1) % max(args.n_directions // 12, 1) == 0
                    or direction_index + 1 == args.n_directions):
                print(f'  support {direction_index + 1}/'
                      f'{args.n_directions}, success='
                      f'{support_success[condition, :direction_index+1].sum()}')

        unique_points, source_indices = unique_points_with_sources(
            support_points[condition], support_success[condition])
        unique_counts[condition] = len(unique_points)
        if len(unique_points) < 3:
            condition_metadata.append({
                'condition_index': condition,
                'status': 'insufficient_unique_support_points',
                'elapsed_seconds': time.perf_counter() - condition_started,
            })
            continue
        try:
            hull = ConvexHull(unique_points)
        except QhullError:
            condition_metadata.append({
                'condition_index': condition,
                'status': 'convex_hull_failed',
                'elapsed_seconds': time.perf_counter() - condition_started,
            })
            continue
        hull_unique_indices = np.asarray(hull.vertices, dtype=int)
        hull_points = unique_points[hull_unique_indices]
        hull_source_indices = source_indices[hull_unique_indices]
        hull_vertex_counts[condition] = len(hull_points)
        support_centroids[condition] = np.mean(hull_points, axis=0)
        pairwise = np.linalg.norm(
            hull_points[:, None, :] - hull_points[None, :, :], axis=2)
        hull_diameters[condition] = float(np.max(pairwise))

        chord_specs = []
        for i in range(len(hull_points)):
            j = (i + 1) % len(hull_points)
            for lam in args.edge_lambdas:
                chord_specs.append(('hull_edge', i, j, lam))
        rng = np.random.RandomState(SEED + 100_000 + condition)
        for i, j in nonadjacent_pairs(
                len(hull_points), args.random_chords, rng):
            for lam in args.random_lambdas:
                chord_specs.append(('nonadjacent_random', i, j, lam))

        print(f'  unique={len(unique_points)}, hull={len(hull_points)}, '
              f'diameter={hull_diameters[condition]:.6g}, '
              f'Check{len(chord_specs)}interior point of string')
        before = len(line_rows)
        for test_index, (line_type, i, j, lam) in enumerate(chord_specs):
            endpoint_1 = hull_points[i]
            endpoint_2 = hull_points[j]
            target = (1.0 - lam) * endpoint_1 + lam * endpoint_2
            projection, distance = project_distance(ec, target)
            normalized_distance = (
                distance / hull_diameters[condition]
                if np.isfinite(distance) and hull_diameters[condition] > 0
                else np.nan)
            line_rows.append({
                'case': casename,
                'condition_index': condition,
                'line_type': line_type,
                'endpoint_index_1': int(hull_source_indices[i]),
                'endpoint_index_2': int(hull_source_indices[j]),
                'lambda': float(lam),
                'endpoint_1_p': float(endpoint_1[0]),
                'endpoint_1_q': float(endpoint_1[1]),
                'endpoint_2_p': float(endpoint_2[0]),
                'endpoint_2_q': float(endpoint_2[1]),
                'target_p': float(target[0]),
                'target_q': float(target[1]),
                'projection_success': projection is not None,
                'projection_p': (float(projection[0])
                                 if projection is not None else np.nan),
                'projection_q': (float(projection[1])
                                 if projection is not None else np.nan),
                'distance_pu': float(distance),
                'condition_hull_diameter_pu': float(
                    hull_diameters[condition]),
                'normalized_distance': float(normalized_distance),
            })
            if ((test_index + 1) % 250 == 0
                    or test_index + 1 == len(chord_specs)):
                current = line_rows[before:]
                successful = sum(row['projection_success'] for row in current)
                print(f'  chord {test_index + 1}/{len(chord_specs)}, '
                      f'Projection successful={successful}/{len(current)}')

        current_rows = line_rows[before:]
        condition_metadata.append({
            'condition_index': condition,
            'status': 'complete',
            'support_success_count': int(support_success[condition].sum()),
            'unique_support_count': int(len(unique_points)),
            'hull_vertex_count': int(len(hull_points)),
            'hull_diameter_pu': float(hull_diameters[condition]),
            'line_test_count': len(current_rows),
            'projection_success_count': sum(
                row['projection_success'] for row in current_rows),
            'max_projection_distance_pu': float(np.nanmax([
                row['distance_pu'] for row in current_rows])),
            'max_normalized_distance': float(np.nanmax([
                row['normalized_distance'] for row in current_rows])),
            'elapsed_seconds': time.perf_counter() - condition_started,
        })
        write_csv(case_dir / 'line_tests.csv', line_rows, LINE_FIELDS)
        np.savez(
            case_dir / 'support_data_partial.npz',
            dthetas=dthetas,
            angles=angles,
            directions=directions,
            support_points=support_points,
            support_success=support_success,
            support_centroids=support_centroids,
            hull_diameters=hull_diameters,
            unique_support_counts=unique_counts,
            hull_vertex_counts=hull_vertex_counts,
        )

    write_csv(case_dir / 'line_tests.csv', line_rows, LINE_FIELDS)
    np.savez(
        case_dir / 'support_data.npz',
        dthetas=dthetas,
        angles=angles,
        directions=directions,
        support_points=support_points,
        support_success=support_success,
        support_centroids=support_centroids,
        hull_diameters=hull_diameters,
        unique_support_counts=unique_counts,
        hull_vertex_counts=hull_vertex_counts,
        elapsed_seconds=np.asarray(time.perf_counter() - case_started),
        interpretation=np.asarray(
            'single-projection finite support/chord proximity diagnostic; '
            'not a strict mathematical convexity proof'),
    )

    completed = [row for row in condition_metadata
                 if row['status'] == 'complete']
    successful_tests = sum(
        row['projection_success'] for row in line_rows)
    finite_distances = np.asarray([
        row['distance_pu'] for row in line_rows
        if np.isfinite(row['distance_pu'])], dtype=float)
    finite_normalized = np.asarray([
        row['normalized_distance'] for row in line_rows
        if np.isfinite(row['normalized_distance'])], dtype=float)
    verdict = 'FINITE_CHORD_PROXIMITY_EVALUATED'
    summary = {
        'case': casename,
        'verdict': verdict,
        'interpretation': (
            'All reported values use one projection solve per tested chord '
            'sample. Distances quantify observed proximity to returned AC-'
            'feasible points; they do not prove strict mathematical convexity.'),
        'tested_conditions': args.n_conditions,
        'completed_conditions': len(completed),
        'support_directions_per_condition': args.n_directions,
        'support_success_rate': float(np.mean(support_success)),
        'total_line_tests': len(line_rows),
        'successful_line_projections': successful_tests,
        'line_projection_success_rate': (
            successful_tests / len(line_rows) if line_rows else np.nan),
        'max_projection_distance_pu': (
            float(np.max(finite_distances)) if finite_distances.size else None),
        'max_normalized_distance': (
            float(np.max(finite_normalized)) if finite_normalized.size else None),
        'max_normalized_distance_percent': (
            100.0 * float(np.max(finite_normalized))
            if finite_normalized.size else None),
        'elapsed_seconds': time.perf_counter() - case_started,
        'condition_metadata': condition_metadata,
    }
    (case_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[{casename}] {verdict}; Projection successful='
          f'{successful_tests}/{len(line_rows)}, '
          f'maximum relative distance={summary["max_normalized_distance_percent"]:.6f}%, '
          f'Time consuming={summary["elapsed_seconds"]:.1f}s')
    return summary


def main():
    global OUT
    parser = argparse.ArgumentParser(
        description='use360Single projection evaluation of support points and chord interior points in 2DTDIdomain approximate convexity')
    parser.add_argument('--case', choices=CASE_NAMES, default=None,
                        help=('Only run the specified system; run in sequence if not specifiedcase33, '
                              'case118, case533'))
    parser.add_argument('--quick', action='store_true',
                        help='1Working conditions, 36direction, 30random strings, check program only')
    parser.add_argument('--n-conditions', type=int, default=N_CONDITIONS)
    parser.add_argument('--n-directions', type=int, default=N_DIRECTIONS)
    parser.add_argument('--random-chords', type=int, default=N_RANDOM_CHORDS)
    args = parser.parse_args()
    args.edge_lambdas = EDGE_LAMBDAS
    args.random_lambdas = RANDOM_LAMBDAS
    if args.quick:
        OUT = OUT / 'quick_smoke'
        args.n_conditions = 1
        args.n_directions = 36
        args.random_chords = 30
    if (args.n_conditions < 1 or args.n_directions < 3
            or args.random_chords < 0):
        raise ValueError('The number of operating conditions, the number of directions or the number of random chords is illegal.')

    OUT.mkdir(parents=True, exist_ok=True)
    selected = (args.case,) if args.case else DEFAULT_CASES
    print('=' * 78)
    print('R2.5 support point convex hull＋Evaluation of approximate convexity of single projection of points inside a chord')
    print(f'mode={"quick" if args.quick else "formal"}; system={list(selected)}')
    print(f'Working conditions={args.n_conditions}; support direction={args.n_directions}; '
          f'random non-adjacent strings={args.random_chords}')

    summaries = [evaluate_case(casename, args, OUT)
                 for casename in selected]
    summary_fields = [
        'case', 'verdict', 'tested_conditions', 'completed_conditions',
        'support_directions_per_condition', 'support_success_rate',
        'total_line_tests', 'successful_line_projections',
        'line_projection_success_rate', 'max_projection_distance_pu',
        'max_normalized_distance', 'max_normalized_distance_percent',
        'elapsed_seconds',
    ]
    summary_path = OUT / 'summary.csv'
    # Formal experiments allow for case-by-case computation, as well as for running three single-phase cases at a time by default. Regardless of
    # For any kind of entry, already completed cases must be merged to prevent the subsequent run from converting previously completed cases.
    # case36 from summary.csv and settings.json Erase from case list.
    if not args.quick and summary_path.exists():
        with summary_path.open('r', newline='', encoding='utf-8-sig') as stream:
            previous = {row['case']: row for row in csv.DictReader(stream)}
        previous.update({row['case']: row for row in summaries})
        summaries = [previous[name] for name in CASE_NAMES
                     if name in previous]
    write_csv(summary_path, summaries, summary_fields)
    settings = vars(args).copy()
    settings['edge_lambdas'] = list(settings['edge_lambdas'])
    settings['random_lambdas'] = list(settings['random_lambdas'])
    settings['requested_cases'] = list(selected)
    settings['selected_cases'] = [row['case'] for row in summaries]
    settings['output_dir'] = str(OUT)
    (OUT / 'settings.json').write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nfinal result: ')
    for summary in summaries:
        print(f'  {summary["case"]}: {summary["verdict"]}, '
              f'max normalized='
              f'{summary["max_normalized_distance_percent"]:.6f}%')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
