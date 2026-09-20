# -*- coding: utf-8 -*-
"""Plot the R1.5/R3.2 downstream-dispatch evidence.

The script prefers the 5-DSO x 5-TSO grid summary.  Before that experiment is
available it can also plot the earlier nominal-DSO TSO-load sweep.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from Simulator import PROJECT_ROOT


DEFAULT_ROOT = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'downstream' / 'case4gs_ts_case33bw_ds'
)
DEFAULT_GRID = DEFAULT_ROOT / 'operating_grid_random_dtheta' / 'grid_summary.csv'
DEFAULT_LEGACY = DEFAULT_ROOT / 'load_scenarios' / 'scenario_summary.csv'


def read_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Result file is empty: {path}')
    numeric = {
        'ts_load_factor', 'dso_load_factor', 'coverage_percent',
        'dso_scenario_index', 'dso_active_load_factor',
        'dso_reactive_load_factor',
        'coverage_percentile', 'coverage_rank_zero_based',
        'undercoverage_percent', 'overcoverage_percent', 'base_cost',
        'pinn_cost', 'true_cost', 'pinn_true_cost_gap',
        'pinn_true_relative_cost_gap_percent', 'value_retention_percent',
        'pinn_true_pg_l1_mw',
    }
    for row in rows:
        for key in numeric & row.keys():
            row[key] = float(row[key])
        row.setdefault('dso_load_factor', 1.0)
    return rows


def dso_group_field(rows):
    if 'coverage_percentile' in rows[0]:
        return 'coverage_percentile'
    if 'dso_scenario_index' in rows[0]:
        return 'dso_scenario_index'
    return 'dso_load_factor'


def dso_label(field, value):
    if field == 'coverage_percentile':
        return f'P{int(value):02d}'
    return f'Sample {int(value) + 1}' if field == 'dso_scenario_index' else f'{value:.1f}'


def save_figure(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f'{stem}.png'
    svg = out_dir / f'{stem}.svg'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {png}')
    print(f'saved: {svg}')


def nominal_cost_plot(rows, out_dir):
    group = dso_group_field(rows)
    dso_values = sorted({row[group] for row in rows})
    if group == 'coverage_percentile':
        nominal_dso = min(dso_values, key=lambda value: abs(value - 50.0))
    elif group == 'dso_scenario_index':
        nominal_dso = dso_values[0]
    else:
        nominal_dso = min(dso_values, key=lambda value: abs(value - 1.0))
    subset = sorted(
        (row for row in rows if abs(row[group] - nominal_dso) < 1e-9),
        key=lambda row: row['ts_load_factor'],
    )
    x = np.asarray([row['ts_load_factor'] for row in subset])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    colors = {'base_cost': '#6b7280', 'pinn_cost': '#2563eb', 'true_cost': '#dc2626'}
    labels = {'base_cost': 'No flexibility', 'pinn_cost': 'PINN region',
              'true_cost': 'Reference region'}
    for key in ('base_cost', 'pinn_cost', 'true_cost'):
        axes[0].plot(x, [row[key] for row in subset], marker='o', linewidth=1.8,
                     color=colors[key], label=labels[key])
    axes[0].set_xlabel('TSO load factor')
    axes[0].set_ylabel('Dispatch cost')
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title(f'DSO {dso_label(group, nominal_dso)}')

    axes[1].plot(
        x, [row['value_retention_percent'] for row in subset],
        marker='o', color='#059669', linewidth=1.8, label='Value retention',
    )
    axes[1].set_xlabel('TSO load factor')
    axes[1].set_ylabel('Value retention (%)', color='#059669')
    axes[1].tick_params(axis='y', labelcolor='#059669')
    axes[1].grid(alpha=0.25)
    right = axes[1].twinx()
    right.plot(
        x, [row['pinn_true_relative_cost_gap_percent'] for row in subset],
        marker='s', color='#7c3aed', linewidth=1.8,
        label='Relative cost gap',
    )
    right.set_ylabel('PINN-to-reference cost gap (%)', color='#7c3aed')
    right.tick_params(axis='y', labelcolor='#7c3aed')
    axes[1].set_title('Economic impact of approximation')
    fig.tight_layout()
    save_figure(fig, out_dir, 'downstream_nominal_cost')


def _grid(rows, key, dso_values, tso_values, group):
    lookup = {(row[group], row['ts_load_factor']): row for row in rows}
    return np.asarray([
        [lookup[(dso, tso)][key] for tso in tso_values]
        for dso in dso_values
    ], dtype=float)


def grid_heatmaps(rows, out_dir):
    group = dso_group_field(rows)
    dso_values = sorted({row[group] for row in rows})
    tso_values = sorted({row['ts_load_factor'] for row in rows})
    if len(dso_values) < 2:
        return
    panels = (
        ('value_retention_percent', 'Value retention (%)', 'YlGnBu'),
        ('pinn_true_relative_cost_gap_percent', 'Relative cost gap (%)', 'YlOrRd'),
        ('pinn_true_pg_l1_mw', 'Active dispatch deviation (MW)', 'PuBu'),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 4.1), constrained_layout=True)
    for ax, (key, title, cmap) in zip(axes, panels):
        values = _grid(rows, key, dso_values, tso_values, group)
        image = ax.imshow(values, origin='lower', aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(tso_values)), [f'{v:.1f}' for v in tso_values])
        ax.set_yticks(range(len(dso_values)), [dso_label(group, v) for v in dso_values])
        ax.set_xlabel('TSO load factor')
        ax.set_ylabel('DSO operating condition')
        ax.set_title(title)
        for i in range(len(dso_values)):
            for j in range(len(tso_values)):
                ax.text(j, i, f'{values[i, j]:.3g}', ha='center', va='center',
                        fontsize=8, color='black')
        fig.colorbar(image, ax=ax, shrink=0.88)
    save_figure(fig, out_dir, 'downstream_grid_heatmaps')


def coverage_cost_plot(rows, out_dir):
    if 'coverage_percent' not in rows[0]:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    group = dso_group_field(rows)
    dso_values = sorted({row[group] for row in rows})
    colors = plt.cm.viridis(np.linspace(0.12, 0.90, len(dso_values)))
    for dso, color in zip(dso_values, colors):
        subset = [row for row in rows if abs(row[group] - dso) < 1e-9]
        ax.scatter(
            [row['undercoverage_percent'] for row in subset],
            [row['pinn_true_relative_cost_gap_percent'] for row in subset],
            s=45, color=color, label=dso_label(group, dso), alpha=0.85,
        )
    ax.set_xlabel('Mean radial undercoverage (%)')
    ax.set_ylabel('PINN-to-reference cost gap (%)')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    # This is a scenario distribution, not a regression or a claim that
    # mean radial undercoverage has a monotonic causal relation with cost.
    ax.set_title('Scenario-wise undercoverage and downstream cost gap')
    save_figure(fig, out_dir, 'coverage_cost_relationship')


def main():
    parser = argparse.ArgumentParser(description='Plot R1.5/R3.2 downstream results')
    parser.add_argument('--data', type=Path, default=None)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    data = args.data
    if data is None:
        data = DEFAULT_GRID if DEFAULT_GRID.exists() else DEFAULT_LEGACY
    if not data.exists():
        raise FileNotFoundError(
            f'No downstream experimental data found: {data}\n'
            'Please run first main_revise1_5and3_2downstream.py.'
        )
    out_dir = args.out or data.parent / 'figures'
    rows = read_rows(data)
    nominal_cost_plot(rows, out_dir)
    grid_heatmaps(rows, out_dir)
    coverage_cost_plot(rows, out_dir)


if __name__ == '__main__':
    main()
