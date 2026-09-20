# -*- coding: utf-8 -*-
"""Plot existing R2.5 support/chord caches to illustrate near-convexity.

This is a plotting-only postprocessor: it does not rerun power flow,
directional support optimization, or projection optimization.  The reported
distance is the raw Euclidean distance from a tested convex-hull chord sample
to the AC-feasible point returned by the projection solve.  Dividing this
distance by the support-point hull diameter makes the four systems comparable.

Outputs
-------
``near_convexity_global_and_local.{png,pdf}``
    Four global hull/projection overlays, each paired with a separate local
    panel of its largest normalized observed distance.
``near_convexity_summary.csv``
    Complete sample counts and the maximum raw/normalized observed distances.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.spatial import ConvexHull

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Simulator import PROJECT_ROOT


RESULT_ROOT = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'support_chord_convexity'
)
CASES = (
    ('case33bw_ds', '33-bus system'),
    ('case118zh_ds', '118-bus system'),
    ('case533mt_hi_ds', '533-bus system'),
    ('case36real_3phase_ds', '36-bus three-phase system'),
)
OUTPUT_STEM = 'near_convexity_global_and_local'


def read_rows(path: Path):
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def as_bool(value):
    return str(value).strip().lower() == 'true'


def hull_geometry(points, success):
    points = np.asarray(points, dtype=float)
    success = np.asarray(success, dtype=bool)
    valid = points[success & np.all(np.isfinite(points), axis=1)]
    hull = ConvexHull(valid)
    vertices = valid[hull.vertices]
    closed = np.vstack((vertices, vertices[0]))
    pairwise = np.linalg.norm(
        vertices[:, None, :] - vertices[None, :, :], axis=2)
    diameter = float(np.max(pairwise))
    return valid, vertices, closed, diameter


def finite_projection_rows(rows):
    output = []
    for row in rows:
        if not as_bool(row['projection_success']):
            continue
        try:
            values = [float(row[key]) for key in (
                'distance_pu', 'projection_p', 'projection_q')]
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(values)):
            output.append(row)
    return output


def point(row, prefix):
    return np.asarray(
        [float(row[f'{prefix}_p']), float(row[f'{prefix}_q'])], dtype=float)


def load_case_result(case_name):
    case_dir = RESULT_ROOT / case_name
    data_path = case_dir / 'support_data.npz'
    rows_path = case_dir / 'line_tests.csv'
    if not data_path.exists() or not rows_path.exists():
        raise FileNotFoundError(
            f'Missing completed R2.5 result for {case_name}: {case_dir}')

    with np.load(data_path) as saved:
        support_points = np.asarray(saved['support_points'], dtype=float)
        support_success = np.asarray(saved['support_success'], dtype=bool)
        directions = np.asarray(saved['directions'], dtype=float)
    rows = read_rows(rows_path)
    successful = finite_projection_rows(rows)

    diameters = np.full(len(support_points), np.nan)
    geometries = []
    for condition in range(len(support_points)):
        geometry = hull_geometry(
            support_points[condition], support_success[condition])
        geometries.append(geometry)
        diameters[condition] = geometry[-1]

    if not successful:
        raise RuntimeError(f'{case_name} has no successful chord projections.')
    worst = max(
        successful,
        key=lambda row: (
            float(row['distance_pu'])
            / diameters[int(row['condition_index'])]))
    condition = int(worst['condition_index'])
    max_raw = max(float(row['distance_pu']) for row in successful)
    max_relative = max(
        float(row['distance_pu']) / diameters[int(row['condition_index'])]
        for row in successful)
    return {
        'case': case_name,
        'support_directions': len(directions),
        'support_points': support_points,
        'support_success': support_success,
        'rows': rows,
        'successful_rows': successful,
        'geometries': geometries,
        'diameters': diameters,
        'worst': worst,
        'worst_condition': condition,
        'max_raw_distance': max_raw,
        'max_relative_distance': max_relative,
    }


def plot_case(ax, zoom, display_name, result):
    condition = result['worst_condition']
    valid, _, hull_closed, diameter = result['geometries'][condition]
    rows = [
        row for row in result['successful_rows']
        if int(row['condition_index']) == condition
        and row['line_type'] == 'hull_edge'
    ]
    projections = np.asarray([
        [float(row['projection_p']), float(row['projection_q'])]
        for row in rows
    ])
    worst = result['worst']
    target = point(worst, 'target')
    projection = point(worst, 'projection')
    endpoint_1 = point(worst, 'endpoint_1')
    endpoint_2 = point(worst, 'endpoint_2')
    distance = float(worst['distance_pu'])
    relative = distance / diameter

    ax.plot(hull_closed[:, 0], hull_closed[:, 1], color='black', lw=1.35,
            label='Support-point convex hull')
    ax.scatter(projections[:, 0], projections[:, 1], s=3.5,
               color='tab:orange', alpha=0.45, rasterized=True,
               label='AC-feasible projected samples')
    ax.scatter(valid[:, 0], valid[:, 1], s=8, facecolors='none',
               edgecolors='tab:blue', linewidths=0.55,
               label='AC support points')
    ax.scatter(target[0], target[1], marker='x', color='crimson', s=42,
               linewidths=1.4, zorder=6)
    ax.set_title(
        f'{display_name} (condition {condition + 1})\n'
        f'max normalized distance = {100.0 * relative:.4f}%')
    ax.set_xlabel('TDI active power P (p.u.)')
    ax.set_ylabel('TDI reactive power Q (p.u.)')
    ax.grid(alpha=0.2)
    ax.set_aspect('equal', adjustable='box')

    zoom.plot([endpoint_1[0], endpoint_2[0]],
              [endpoint_1[1], endpoint_2[1]], color='black', lw=1.25)
    if len(projections):
        zoom.scatter(projections[:, 0], projections[:, 1], s=6,
                     color='tab:orange', alpha=0.45, rasterized=True)
    zoom.scatter(target[0], target[1], marker='x', color='crimson',
                 s=52, linewidths=1.7, zorder=7)
    zoom.scatter(projection[0], projection[1], facecolors='none',
                 edgecolors='tab:purple', s=48, linewidths=1.5, zorder=7)
    zoom.plot([target[0], projection[0]], [target[1], projection[1]],
              color='tab:purple', lw=1.25, zorder=6)

    center = 0.5 * (target + projection)
    span = max(12.0 * distance, 0.008 * diameter)
    zoom.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
    zoom.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
    zoom.set_aspect('equal', adjustable='box')
    zoom.grid(alpha=0.18)
    zoom.tick_params(labelsize=6)
    zoom.xaxis.set_major_locator(MaxNLocator(3))
    zoom.yaxis.set_major_locator(MaxNLocator(4))
    zoom.text(
        0.03, 0.97,
        f'$d_{{max}}$={distance:.3e} p.u.\n'
        f'$d_{{max}}/D_\\Omega$={100.0 * relative:.4f}%',
        transform=zoom.transAxes, ha='left', va='top', fontsize=7,
        bbox={'boxstyle': 'round,pad=0.18', 'facecolor': 'white',
              'edgecolor': 'none', 'alpha': 0.82})
    zoom.set_xlabel('P (p.u.)', fontsize=8)
    zoom.set_ylabel('Q (p.u.)', fontsize=8)


def write_summary(results, path):
    fields = [
        'case', 'tested_chord_samples', 'successful_projection_samples',
        'failed_projection_samples', 'support_directions_per_condition',
        'worst_condition_one_based', 'maximum_projection_distance_pu',
        'maximum_normalized_projection_distance',
        'maximum_normalized_projection_distance_percent',
    ]
    rows = []
    for result in results:
        total = len(result['rows'])
        successful = len(result['successful_rows'])
        rows.append({
            'case': result['case'],
            'tested_chord_samples': total,
            'successful_projection_samples': successful,
            'failed_projection_samples': total - successful,
            'support_directions_per_condition': result['support_directions'],
            'worst_condition_one_based': result['worst_condition'] + 1,
            'maximum_projection_distance_pu': result['max_raw_distance'],
            'maximum_normalized_projection_distance': (
                result['max_relative_distance']),
            'maximum_normalized_projection_distance_percent': (
                100.0 * result['max_relative_distance']),
        })
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    results = [load_case_result(case) for case, _ in CASES]
    fig = plt.figure(figsize=(15.2, 8.8))
    grid = fig.add_gridspec(
        2, 6, width_ratios=(1.0, 1.0, 0.72, 1.0, 1.0, 0.72),
        wspace=0.50, hspace=0.42)
    global_axes = []
    for index, ((_, display_name), result) in enumerate(zip(CASES, results)):
        row = index // 2
        start = 0 if index % 2 == 0 else 3
        ax = fig.add_subplot(grid[row, start:start + 2])
        zoom = fig.add_subplot(grid[row, start + 2])
        plot_case(ax, zoom, display_name, result)
        global_axes.append(ax)

    handles, labels = global_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               frameon=False, fontsize=9)
    fig.suptitle(
        'Observed proximity between AC-feasible projections and '
        'support-point convex hulls', fontsize=13)
    fig.subplots_adjust(left=0.060, right=0.985, bottom=0.105, top=0.90)

    png_path = RESULT_ROOT / f'{OUTPUT_STEM}.png'
    pdf_path = RESULT_ROOT / f'{OUTPUT_STEM}.pdf'
    csv_path = RESULT_ROOT / 'near_convexity_summary.csv'
    fig.savefig(png_path, dpi=400, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    write_summary(results, csv_path)

    print(f'Figure (PNG): {png_path}')
    print(f'Figure (PDF): {pdf_path}')
    print(f'Summary:      {csv_path}')
    for result in results:
        total = len(result['rows'])
        success = len(result['successful_rows'])
        print(
            f'{result["case"]}: {success}/{total} projections, '
            f'max distance={result["max_raw_distance"]:.6e} p.u., '
            f'max normalized={100*result["max_relative_distance"]:.6f}%')


if __name__ == '__main__':
    main()
