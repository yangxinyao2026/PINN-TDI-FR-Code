# -*- coding: utf-8 -*-
"""Plot the R1.7 zero-shot topology-change results for case33."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Compatible fromPyCharm/Run this file directly from the command line.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Simulator import PROJECT_ROOT


DEFAULT_ROOT = (PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' /
                'comparison' / 'topology' / 'case33bw_ds')
TOPOLOGIES = ('Base', 'T1', 'T2', 'T3', 'T4', 'T5')


def finite(values):
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def ordered(points):
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return points
    center = np.mean(points, axis=0)
    angle = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angle)]
    return np.vstack((points, points[0]))


def polygon_vertices(A, b, tol=1e-7):
    vertices = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            matrix = np.vstack((A[i], A[j]))
            if abs(np.linalg.det(matrix)) <= 1e-10:
                continue
            point = np.linalg.solve(matrix, np.array([b[i], b[j]]))
            if np.all(A @ point <= b + tol):
                vertices.append(point)
    if not vertices:
        return np.empty((0, 2))
    return ordered(np.unique(np.round(np.asarray(vertices), 10), axis=0))


def main(root=None):
    root = Path(root) if root is not None else DEFAULT_ROOT
    summary_path = root / 'summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(
            f'not found{summary_path}, Please run firstmain_revise1_7_topology.py')
    with open(summary_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    row_map = {row['topology']: row for row in rows}
    data = {
        name: np.load(root / name / 'evaluation_data.npz', allow_pickle=True)
        for name in TOPOLOGIES}

    colors = plt.cm.tab10(np.linspace(0.0, 0.75, len(TOPOLOGIES)))
    fig, compact_axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    # Preserve compatibility with the original two-dimensional axis indexing.
    axes = np.asarray([
        [compact_axes[0], compact_axes[1]],
        [compact_axes[0], compact_axes[1]],
    ], dtype=object)

    # Panel (a): compare topology-specific true regions at the nominal condition.
    ax = axes[0, 0]
    for color, name in zip(colors, TOPOLOGIES):
        boundary = ordered(data[name]['true_support'][0])
        if len(boundary):
            ax.plot(boundary[:, 0], boundary[:, 1], color=color,
                    linewidth=1.5, label=f'True {name}')
    learned = polygon_vertices(
        data['Base']['predicted_A'][0], data['Base']['predicted_b'][0])
    if len(learned):
        ax.plot(learned[:, 0], learned[:, 1], 'k--', linewidth=2.0,
                label='Base-trained PINN')
    ax.set_xlabel('TDI active power $P$ (p.u.)')
    ax.set_ylabel('TDI reactive power $Q$ (p.u.)')
    ax.set_title('(a) Nominal regions under radial reconfiguration')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    # (b) rhodistribution; dashed line1Distinguish between radial under-coverage and over-coverage.
    ax = axes[0, 1]
    rho_data = [finite(data[name]['coverage']) for name in TOPOLOGIES]
    ax.boxplot(
        rho_data, tick_labels=TOPOLOGIES, showfliers=False, widths=0.58
    )
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1.0)
    ax.set_ylabel('Coverage ratio $k_P/k_\\Omega$')
    ax.set_title('(b) Zero-shot coverage distribution')
    ax.grid(axis='y', alpha=0.25)

    # (c) Feasibility and Optimality of Squared ErrorsP95, logAxes show different magnitudes.
    ax = axes[1, 0]
    # (d) Coverage tail and over/Undercoverage decomposition. Different topologies use their own reference points, so they are not used here.
    # k_omegaDirect subtraction defines cross-topology “area offsets“ to avoid mixing in reference point position changes.
    ax = axes[1, 1]
    fig.tight_layout()
    png = root / 'topology_comparison.png'
    pdf = root / 'topology_comparison.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {png}')
    print(f'saved: {pdf}')


if __name__ == '__main__':
    main()
