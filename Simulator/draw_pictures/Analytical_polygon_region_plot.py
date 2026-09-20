# -*- coding: utf-8 -*-
"""Plot analytical-polygon and neural-network feasible-region comparisons."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from Simulator import PROJECT_ROOT
import Simulator.cases.fixed_module2 as fixed_module2


def get_weights_dir(casename):
    """Get the weight file directory (with Analytical_polygon.py consistent) """
    return (f"{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\"
            "A(8,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)")


# =============================================================================
# Helper function
# =============================================================================

def polygon_from_Ab(A, b):
    """from Ax <= b Generate polygon vertices"""
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
# Drawing
# =============================================================================

def plot_comparison(data_path, output_dir):
    """Read and plot comparison data"""
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
# main program
# =============================================================================

if __name__ == '__main__':
    case_dirs = {
        # 'case10ba_ds': get_weights_dir('case10ba_ds'),
         'case33bw_ds': get_weights_dir('case33bw_ds'),
        # 'case118zh_ds': get_weights_dir('case118zh_ds'),
        # 'case533mt_hi_ds': get_weights_dir('case533mt_hi_ds'),
    }

    for casename, result_dir in case_dirs.items():
        print(f'\nDrawing: {casename}')

        comparison_dir = (f'{result_dir}\\figures\\comparison\\feasible\\contrast\\'
                          'comparison_AnalyticalPolygon\\')

        data_path = os.path.join(comparison_dir, 'analytical_polygon_region_comparison_data.npz')
        if os.path.exists(data_path):
            plot_comparison(data_path, comparison_dir)
        else:
            print(f"Data file not found: {data_path}")
            print("Please run first Analytical_polygon.py Calculated data")
