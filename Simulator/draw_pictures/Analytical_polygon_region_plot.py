# -*- coding: utf-8 -*-
"""
Analytical_polygon_plot.py
功能：读取 Analytical_polygon.py 保存的数据文件，绘制解析多边形法 vs NN方法对比图

使用方法：先运行 Analytical_polygon.py 计算并保存数据，再运行本脚本绘图
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from Simulator import PROJECT_ROOT
import Simulator.cases.fixed_module2 as fixed_module2


def get_weights_dir(casename):
    """获取权重文件目录（与 Analytical_polygon.py 一致）"""
    return (f"{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\"
            "A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)")


# =============================================================================
# 辅助函数
# =============================================================================

def polygon_from_Ab(A, b):
    """从 Ax <= b 生成多边形顶点"""
    if A is None or b is None or len(A) == 0:
        return None

    vertices = []
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                x, y = np.linalg.solve(np.array([A[i], A[j]]), np.array([b[i], b[j]]))
                if np.all(A @ np.array([x, y]) <= b + 1e-5):
                    vertices.append((x, y))
            except np.linalg.LinAlgError:
                pass

    if not vertices:
        return None

    vertices = np.array(vertices)
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return vertices[sorted_indices]


# =============================================================================
# 绘图
# =============================================================================

def plot_comparison(data_path, output_dir):
    """读取对比数据并绘图"""
    data = np.load(data_path)

    polygon_boundary = data['polygon_boundary']
    n_directions = int(data['n_directions'])
    original_boundary = data['original_boundary']
    fullnet_A = data['fullnet_A']
    fullnet_b = data['fullnet_b']
    xlim = data['xlim']
    ylim = data['ylim']

    plotter = fixed_module2.ShapeDrawer_2D()
    plotter.fig.set_size_inches(8, 8)

    # Original region
    plotter.plot_polygon(x_org=original_boundary,
                         xlim=xlim, ylim=ylim, facecolor='blue', alpha=0.3,
                         label='Original region')

    # Analytical polygon
    ap_patch = MplPolygon(polygon_boundary, closed=True,
                          fill=True, alpha=0.3, color='green',
                          label=f'Analytical polygon (m={n_directions})')
    plotter.ax.add_patch(ap_patch)

    # FullNet Learned region
    plotter.plot_polygon(A=fullnet_A, b=fullnet_b,
                         xlim=xlim, ylim=ylim, facecolor='red', alpha=0.3,
                         label='Learned region')

    plotter.ax.set_xlim(xlim)
    plotter.ax.set_ylim(ylim)

    save_path = os.path.join(output_dir, 'analytical_polygon_region_comparison.svg')
    plotter.save(save_path, dpi=300, format='svg', show_legend=True)
    plt.close()


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    case_dirs = {
        # 'case10ba_ds': get_weights_dir('case10ba_ds'),
         'case33bw_ds': get_weights_dir('case33bw_ds'),
        # 'case118zh_ds': get_weights_dir('case118zh_ds'),
        #  'case533mt_hi_ds': get_weights_dir('case533mt_hi_ds'),
        # 'case36real_3phase_ds': f'{PROJECT_ROOT}\\results\\ds_proj_paper\\case36real_3phase_ds\\A(8,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)',
    }

    for casename, result_dir in case_dirs.items():
        print(f'\n绘图: {casename}')

        comparison_dir = (f'{result_dir}\\figures\\comparison\\feasible\\contrast\\'
                          'comparison_AnalyticalPolygon\\')

        data_path = os.path.join(comparison_dir, 'analytical_polygon_region_comparison_data.npz')
        if os.path.exists(data_path):
            plot_comparison(data_path, comparison_dir)
        else:
            print(f"未找到数据文件: {data_path}")
            print("请先运行 Analytical_polygon.py 计算数据")
