# -*- coding: utf-8 -*-
"""Plot the nominal R1.3 feasible regions with and without thermal constraints."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from Simulator import PROJECT_ROOT


DEFAULT_DATA = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'thermal' / 'case33bw_ds'
    / 'preview_thermal_120pct'
    / 'true_region_preview_data.npz'
)


def closed(points):
    points = np.asarray(points, dtype=float)
    return np.vstack((points, points[0])) if len(points) else points


def plot_preview(data_path=DEFAULT_DATA):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f'Preview data not found: {data_path}\n'
            'Please run first main_revise1_3_thermal.py --stage preview --margin 1.2')
    data = np.load(data_path)
    baseline = np.asarray(data['baseline_boundary'], dtype=float)
    thermal = np.asarray(data['thermal_boundary'], dtype=float)
    active = np.asarray(data['thermal_active'], dtype=bool)
    reference = np.asarray(data['reference_point'], dtype=float)
    margin = float(data['margin'])
    reduction = float(data['area_reduction'])
    if len(baseline) < 3 or len(thermal) < 3:
        raise RuntimeError('The effective boundary points are less than3, unable to draw the feasible region')

    fig, ax = plt.subplots(figsize=(7.3, 6.2))
    base_closed = closed(baseline)
    thermal_closed = closed(thermal)
    ax.fill(base_closed[:, 0], base_closed[:, 1], color='#4C78A8',
            alpha=0.14)
    ax.plot(base_closed[:, 0], base_closed[:, 1], color='#4C78A8',
            lw=2.0, label='Baseline true region')
    ax.fill(thermal_closed[:, 0], thermal_closed[:, 1], color='#F28E2B',
            alpha=0.22)
    ax.plot(thermal_closed[:, 0], thermal_closed[:, 1], color='#E15759',
            lw=2.1, label=fr'Thermal-constrained true region '
                          fr'($I_{{max}}={margin:.2f}I_{{nom}}$)')
    if len(active) == len(thermal) and np.any(active):
        ax.scatter(thermal[active, 0], thermal[active, 1], s=22,
                   color='#B22222', edgecolors='white', linewidths=0.35,
                   zorder=4, label='At least one line utilization >=99%')
    if np.all(np.isfinite(reference)):
        ax.scatter(reference[0], reference[1], marker='*', s=150,
                   color='black', zorder=5, label='Nominal reference point')

    ax.set_xlabel(r'TDI active power $P$ (p.u.)')
    ax.set_ylabel(r'TDI reactive power $Q$ (p.u.)')
    ax.set_title('Impact of branch thermal limits on the true flexibility region')
    ax.text(0.02, 0.02, f'Area reduction: {100*reduction:.2f}%',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                      edgecolor='#BBBBBB'))
    ax.grid(alpha=0.24)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=9, frameon=True)
    fig.tight_layout()

    out = data_path.parent
    png = out / 'true_region_preview.png'
    pdf = out / 'true_region_preview.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'[preview plot] PNG: {png}')
    print(f'[preview plot] PDF: {pdf}')
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description='R1.3Thermal Constrained Real Flexible Domain Preview')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    plot_preview(args.data)


if __name__ == '__main__':
    main()
