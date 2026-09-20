# -*- coding: utf-8 -*-
"""Plot analytical-polygon and neural-network error comparisons."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt

from Simulator import PROJECT_ROOT
from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase

# ============ Font configuration (IEEE Transactions: Times New Roman + STIX)  ============
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# Helper function
# =============================================================================

def get_case_config(casename):
    """Get the weight directory configuration corresponding to the calculation example"""
    is_3phase = '3phase' in casename

    if is_3phase:
        configs = {
            8: 'A(8,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)',
            36: 'A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)',
        }
    elif '533mt' in casename:
        configs = {
            8: 'A(8,2)_type3(9-36)_lr1(1e-4)_lr2(1e-4)_rate(1e-4)',
            36: 'A(36,2)_type3(9-36)_lr1(1e-4)_lr2(1e-4)_rate(1e-4)',
        }
    elif '33bw' in casename:
        configs = {
            8: 'A(8,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)',
            36: 'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)',
        }
    else:
        configs = {
            8: 'A(8,2)_type3(97, 107, 109, 80, 63, 31)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)',
            36: 'A(36,2)_type3(97, 107, 109, 80, 63, 31)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)',
        }

    return configs


def plot_boxplot(ax, data, labels, colors, title, ylabel):
    """Draw a boxplot with scatter points"""
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    widths=0.5, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.3, s=8, color=colors[i],
                   edgecolors='black', linewidths=0.3)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=10)


def plot_error_comparison(ap_data_path, nn_data_path, output_dir, casename, m):
    """Read error data and draw feasibility and optimality comparison charts

    Args:
        ap_data_path: Analytical polygon error data path (analytical_polygon_error_data_m{m}.npz)
        nn_data_path: NN feasible error data path (error_analysis_data.npz)
        output_dir: Image output directory
        casename: Case name
        m: Number of directions
    """
    ap_data = np.load(ap_data_path)
    nn_data = np.load(nn_data_path)

    ap_feas = ap_data['analytical_feas']
    ap_opt = ap_data['analytical_opt']
    nn_feas = nn_data['feasible_feas']
    nn_opt = nn_data['feasible_opt']

    labels = [f'NN feasible\n(m={m})', f'Analytical polygon\n(m={m})']
    colors = ['lightcoral', 'lightgreen']

    # ---- Figure1: Feasibility error comparison ----
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 6))
    plot_boxplot(ax1, [nn_feas, ap_feas], labels, colors,
                 f'Feasibility Errors — {casename} (m={m})', 'Feasibility Error')
    fig1.tight_layout()
    feas_path = os.path.join(output_dir, f'feasibility_error_comparison_m{m}.svg')
    fig1.savefig(feas_path, dpi=300, bbox_inches='tight', format='svg')
    feas_path_png = os.path.join(output_dir, f'feasibility_error_comparison_m{m}.png')
    fig1.savefig(feas_path_png, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig1)
    print(f"The feasibility error comparison chart has been saved to: {feas_path}")

    # ---- Figure2: Optimality error comparison ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    plot_boxplot(ax2, [nn_opt, ap_opt], labels, colors,
                 f'Optimality Errors — {casename} (m={m})', 'Optimality Error')
    fig2.tight_layout()
    opt_path = os.path.join(output_dir, f'optimality_error_comparison_m{m}.svg')
    fig2.savefig(opt_path, dpi=300, bbox_inches='tight', format='svg')
    opt_path_png = os.path.join(output_dir, f'optimality_error_comparison_m{m}.png')
    fig2.savefig(opt_path_png, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig2)
    print(f"Optimality error comparison chart has been saved to: {opt_path}")


# =============================================================================
# main program
# =============================================================================

if __name__ == '__main__':
    dscases = {
        # 'case118zh_ds': TD_case.case118zh_ds(),
         'case533mt_hi_ds': TD_case.case533mt_hi_ds(),
        # 'case36real_3phase_ds': DS_case_3phase.case36real_3phase_ds(),
        # 'case33bw_ds': TD_case.case33bw_ds(),
    }

    m_values = [8, 36]

    for casename, ppc in dscases.items():
        configs = get_case_config(casename)
        for m in m_values:
            config_str = configs[m]
            result_dir = f"{PROJECT_ROOT}/results/ds_proj_paper/{casename}/{config_str}"

            # Analytical polygon error data
            ap_dir = os.path.join(result_dir, 'figures', 'comparison', 'feasible', 'contrast', 'comparison_AnalyticalPolygon')
            ap_path = os.path.join(ap_dir, f'analytical_polygon_error_data_m{m}.npz')

            # NN feasible Error data (from Box_error.py)
            nn_path = os.path.join(result_dir, 'figures', 'comparison', 'feasible',
                                   'contrast', 'error_distributions', 'error_analysis_data.npz')

            if not os.path.exists(ap_path):
                print(f"not found Analytical polygon data: {ap_path}")
                print("Please run first Analytical_polygon_error.py Calculated data")
                continue
            if not os.path.exists(nn_path):
                print(f"not found NN feasible data: {nn_path}")
                print("Please run first Box_error.py Calculated data")
                continue

            print(f'\nDrawing: {casename} m={m}')
            plot_error_comparison(ap_path, nn_path, ap_dir, casename, m)
