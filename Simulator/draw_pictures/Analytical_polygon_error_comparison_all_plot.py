# -*- coding: utf-8 -*-
"""
Analytical_polygon_error_comparison_all_plot.py
功能：将各算例的Analytical polygon最优性/可行性误差箱型图画在一起

分两张图（各为单轴）：
1. 可行性误差：每个case对比 m=8 vs m=36
2. 最优性误差：每个case对比 m=8 vs m=36
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from Simulator import PROJECT_ROOT

# ============ 字体配置（IEEE Transactions: Times New Roman + STIX） ============
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# ============ 算例配置 ============
CASES = [
    {
        'casename': 'case33bw_ds',
        'label': '33-bus',
        'configs': {
            8: 'A(8,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)',
            36: 'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)',
        }
    },
    {
        'casename': 'case118zh_ds',
        'label': '118-bus',
        'configs': {
            8: 'A(8,2)_type3(97, 107, 109, 80, 63, 31)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)',
            36: 'A(36,2)_type3(97, 107, 109, 80, 63, 31)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)',
        }
    },
    {
        'casename': 'case533mt_hi_ds',
        'label': '533-bus',
        'configs': {
            8: 'A(8,2)_type3(9-36)_lr1(1e-4)_lr2(1e-4)_rate(1e-4)',
            36: 'A(36,2)_type3(9-36)_lr1(1e-4)_lr2(1e-4)_rate(1e-4)',
        }
    },
    {
        'casename': 'case36real_3phase_ds',
        'label': '36-bus(3ph)',
        'configs': {
            8: 'A(8,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)',
            36: 'A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)',
        }
    },
]


# =============================================================================
# 辅助函数
# =============================================================================

def load_ap_errors(casename, config_str, m):
    """加载Analytical polygon的误差数据"""
    result_dir = f"{PROJECT_ROOT}/results/ds_proj_paper/{casename}/{config_str}"
    ap_path = os.path.join(result_dir, 'figures', 'comparison', 'feasible',
                           'contrast', 'comparison_AnalyticalPolygon',
                           f'analytical_polygon_error_data_m{m}.npz')
    ap_data = np.load(ap_path)
    return ap_data['analytical_feas'], ap_data['analytical_opt']


def plot_grouped_boxplot(ax, data_list, group_labels, ylabel):
    """绘制分组箱线图（m=8和m=36交替排列）

    Args:
        data_list: [m8_case1, m36_case1, m8_case2, m36_case2, ...]
        group_labels: 各case的标签
    """
    n_groups = len(group_labels)
    color_m8 = 'lightblue'
    color_m36 = 'lightgreen'

    bp = ax.boxplot(data_list, patch_artist=True, widths=0.6, showfliers=False)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(color_m8 if i % 2 == 0 else color_m36)
        patch.set_alpha(0.6)

    for i, d in enumerate(data_list):
        x = np.random.normal(i + 1, 0.03, size=len(d))
        color = color_m8 if i % 2 == 0 else color_m36
        ax.scatter(x, d, alpha=0.3, s=6, color=color,
                   edgecolors='black', linewidths=0.3)

    # x轴标签：在每组中间显示case名
    tick_positions = [i * 2 + 1.5 for i in range(n_groups)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels, fontsize=10)

    # 组分隔线
    for i in range(1, n_groups):
        ax.axvline(x=i * 2 + 0.5, color='gray', linestyle='--', alpha=0.3)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [Patch(facecolor=color_m8, alpha=0.6, label='m=8'),
                       Patch(facecolor=color_m36, alpha=0.6, label='m=36')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)


# =============================================================================
# 绘图
# =============================================================================

def plot_all_comparison():
    """绘制所有算例的Analytical polygon误差对比图"""

    all_feas, all_opt = [], []
    group_labels = []

    for case in CASES:
        casename = case['casename']
        label = case['label']
        configs = case['configs']

        for m in [8, 36]:
            feas, opt = load_ap_errors(casename, configs[m], m)
            all_feas.append(feas)
            all_opt.append(opt)
        group_labels.append(label)

    # ---- 图1: 可行性误差 ----
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    plot_grouped_boxplot(ax1, all_feas, group_labels, 'Feasibility Error')
    fig1.tight_layout()

    # ---- 图2: 最优性误差 ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    plot_grouped_boxplot(ax2, all_opt, group_labels, 'Optimality Error')
    fig2.tight_layout()

    output_dir = os.path.join(PROJECT_ROOT, 'results', 'ds_proj_paper')
    os.makedirs(output_dir, exist_ok=True)

    for fig, name in [(fig1, 'ap_feasibility_error_comparison_all'),
                      (fig2, 'ap_optimality_error_comparison_all')]:
        svg_path = os.path.join(output_dir, f'{name}.svg')
        png_path = os.path.join(output_dir, f'{name}.png')
        fig.savefig(svg_path, dpi=300, bbox_inches='tight', format='svg')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {svg_path}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    plot_all_comparison()
