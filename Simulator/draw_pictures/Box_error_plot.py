# -*- coding: utf-8 -*-
"""
Box_error_plot.py
功能：读取 Box_error.py 保存的数据文件，绘制可视化图表

使用方法：先运行 Box_error.py 计算并保存数据，再运行本脚本绘图
"""

import os

from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import Simulator.cases.fixed_module2 as fixed_module2

from Simulator import PROJECT_ROOT


# =============================================================================
# 可行域对比图
# =============================================================================

def plot_feasible_region(data_path, output_dir):
    """读取可行域数据并绘制对比图"""
    data = np.load(data_path)

    boundary_points = data['boundary_points']
    A_moderate = data['A_moderate']
    b_moderate = data['b_moderate']
    A_feasible = data['A_feasible']
    b_feasible = data['b_feasible']
    xlim = data['xlim']
    ylim = data['ylim']
    dtheta = data['dtheta']
    n_directions = int(data['n_directions'])

    plotter = fixed_module2.ShapeDrawer_2D()

    # 绘制原始可行域
    plotter.plot_polygon(x_org=boundary_points, facecolor='blue',
                         xlim=xlim, ylim=ylim, label='Original region')

    # 绘制 moderate 模型
    plotter.plot_polygon(A=A_moderate, b=b_moderate,
                         facecolor='yellow', xlim=xlim, ylim=ylim, label='Learned region-moderate')

    # 绘制 feasible 模型
    plotter.plot_polygon(A=A_feasible, b=b_feasible,
                         facecolor='red', xlim=xlim, ylim=ylim, label='Learned region-feasible')

    # 保存
    save_name = f'{n_directions}_dtheta_{dtheta[0]:.2e}_{dtheta[len(dtheta)//2]:.2e}'
    plotter.save(os.path.join(output_dir, save_name), format='png')
    print(f"可行域对比图已保存至: {output_dir}")


# =============================================================================
# 误差分布图
# =============================================================================

def plot_boxplot(ax, data, labels, colors, title, ylabel):
    """绘制带散点的箱线图"""
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=10, color=colors[i],
                   edgecolors='black', linewidths=0.5)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)


def plot_error_distribution(data_path, output_dir):
    """读取误差分析数据并绘制箱线图"""
    data = np.load(data_path)

    moderate_feas = data['moderate_feas']
    moderate_opt = data['moderate_opt']
    feasible_feas = data['feasible_feas']
    feasible_opt = data['feasible_opt']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = ['Learned region-moderate', 'Learned region-feasible']
    colors = ['lightblue', 'lightcoral']

    # 左图：可行性误差
    plot_boxplot(axes[0], [moderate_feas, feasible_feas], labels, colors,
                 'Feasibility Errors', 'Feasibility Error')

    # 右图：最优性误差
    plot_boxplot(axes[1], [moderate_opt, feasible_opt], labels, colors,
                 'Optimality Errors', 'Optimality Error')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'combined_error_distribution.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
    print(f"合并误差分布图已保存至: {save_path}")
    plt.close()


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    # 定义案例字典
    dscases = {
        # 'case10ba_ds': TD_case.case10ba_ds(),
        # 'case17me_ds': TD_case.case17me_ds(),
        # 'case33bw_ds': TD_case.case33bw_ds(),
        # 'case51ga_ds': TD_case.case51ga_ds(),
        # 'case74_ds': TD_case.case74_ds(),
         'case118zh_ds': TD_case.case118zh_ds(),
        # 'case136ma_ds': TD_case.case136ma_ds(),
        # 'case533mt_hi_ds': TD_case.case533mt_hi_ds(),
        # 'case36real_3phase_ds': DS_case_3phase.case36real_3phase_ds(),
    }
    for casename, ppc in dscases.items():
        is_3phase = '3phase' in casename
        if is_3phase:
            result_dir = f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)'
        else:
            result_dir = (f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\'
                         'A(8,2)_type3(97, 107, 109, 80, 63, 31)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)2')
        output_dir = f'{result_dir}\\figures\\comparison\\feasible\\contrast\\'

        # 读取并绘制可行域对比图
        feasible_data_path = os.path.join(output_dir, 'feasible_region_data.npz')
        if os.path.exists(feasible_data_path):
            plot_feasible_region(feasible_data_path, output_dir)
        else:
            print(f"未找到可行域数据文件: {feasible_data_path}")
            print("请先运行 Box_error.py 计算数据")

        # 读取并绘制误差分布图
        error_dist_dir = os.path.join(os.path.dirname(output_dir), 'error_distributions')
        error_data_path = os.path.join(error_dist_dir, 'error_analysis_data.npz')
        if os.path.exists(error_data_path):
            plot_error_distribution(error_data_path, error_dist_dir)
        else:
            print(f"未找到误差分析数据文件: {error_data_path}")
            print("请先运行 Box_error.py 计算数据")
