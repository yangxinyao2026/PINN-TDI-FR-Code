# -*- coding: utf-8 -*-
"""Plot feasible-region comparisons under parameter perturbations."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from Simulator import PROJECT_ROOT
from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase

# ============ Font configuration (IEEE Transactions: Times New Roman + STIX)  ============
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Helper function
# ============================================================

def polygon_from_Ab(A, b):
    """from Ax <= b Compute ordered polygon vertices"""
    if A is None or b is None or len(A) == 0:
        return None
    vertices = []
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                xy = np.linalg.solve(np.array([A[i], A[j]]), np.array([b[i], b[j]]))
                if np.all(A @ xy <= b + 1e-5):
                    vertices.append(xy)
            except np.linalg.LinAlgError:
                pass
    if not vertices:
        return None
    vertices = np.array(vertices)
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return vertices[np.argsort(angles)]


def plot_polygon_on_ax(ax, vertices, facecolor='blue', alpha=0.2,
                       edgecolor='blue', linewidth=1.0, label=None, linestyle='solid'):
    """Draw a polygon on the specified axis"""
    if vertices is None:
        return
    polygon = MplPolygon(vertices, closed=True, facecolor=facecolor, alpha=alpha,
                          edgecolor=edgecolor, linewidth=linewidth, label=label,
                          linestyle=linestyle)
    ax.add_patch(polygon)
    return polygon


# ============================================================
# Drawing
# ============================================================

def plot_comparison(data_path, output_dir):
    """Read the comparison data and plot it2x3subplot"""
    data = np.load(data_path, allow_pickle=True)
    xlim = data['xlim']
    ylim = data['ylim']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx in range(6):
        ax = axes[idx]
        key = f'scenario_{idx}'
        name = str(data[f'{key}_name'])
        label = str(data[f'{key}_label'])
        true_pts = data[f'{key}_true_pts']
        A_mod = data[f'{key}_A_mod']
        b_mod = data[f'{key}_b_mod']
        A_feas = data[f'{key}_A_feas']
        b_feas = data[f'{key}_b_feas']

        show_label = (idx == 0)

        # Original region (blue)
        plot_polygon_on_ax(ax, true_pts,
                           facecolor='tab:blue', alpha=0.35,
                           edgecolor='darkblue', linewidth=1.5,
                           label='Original region' if show_label else None)

        # Learned region-moderate (orange)
        plot_polygon_on_ax(ax, polygon_from_Ab(A_mod, b_mod),
                           facecolor='tab:orange', alpha=0.35,
                           edgecolor='darkorange', linewidth=1.5,
                           label='Learned region-moderate' if show_label else None)

        # Learned region-feasible (red)
        plot_polygon_on_ax(ax, polygon_from_Ab(A_feas, b_feas),
                           facecolor='tab:red', alpha=0.35,
                           edgecolor='darkred', linewidth=1.5,
                           label='Learned region-feasible' if show_label else None)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'{name} {label}', fontsize=13, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(h_pad=2.0, w_pad=1.5)

    # Shared legend for the whole picture
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=13, frameon=True,
               bbox_to_anchor=(0.5, 1.06), bbox_transform=fig.transFigure,
               ncol=3)

    # save
    out_svg = os.path.join(output_dir, 'region_comparison_6cases.svg')
    out_png = os.path.join(output_dir, 'region_comparison_6cases.png')
    fig.savefig(out_svg, dpi=300, bbox_inches='tight', format='svg')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Image saved:\n  {out_svg}\n  {out_png}")


# ============================================================
# main program
# ============================================================

if __name__ == '__main__':
    # Define cases and corresponding result directories
    CASE_NAME = 'case33bw_ds'
    # CASE_NAME = 'case118zh_ds'
    # CASE_NAME = 'case533mt_hi_ds'
    # CASE_NAME = 'case36real_3phase_ds'
    CONFIG_STR_3PHASE = 'A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)'
    CONFIG_STR_1PHASE = 'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)'

    is_3phase = '3phase' in CASE_NAME
    config_str = CONFIG_STR_3PHASE if is_3phase else CONFIG_STR_1PHASE
    weights_dir = f"{PROJECT_ROOT}/results/ds_proj_paper/{CASE_NAME}/{config_str}"
    comparison_dir = os.path.join(weights_dir, 'figures', '6_dtheta_region')

    data_path = os.path.join(comparison_dir, 'region_comparison_data.npz')
    if os.path.exists(data_path):
        plot_comparison(data_path, comparison_dir)
    else:
        print(f"Data file not found: {data_path}")
        print("Please run first dtheta_region_comparison.py Calculated data")
