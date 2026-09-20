# -*- coding: utf-8 -*-
"""Plot the training and evaluation errors saved by main_ds.py."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Simulator.Plotter import ErrorVisualizer
from Simulator import PROJECT_ROOT


def plot_training_error(data_path, casename, model_type):
    """Read the training error data and draw a box plot"""
    data = np.load(data_path)

    # rebuild ErrorVisualizer of error_history
    visualizer = ErrorVisualizer()
    visualizer.error_history['iterations'] = data['iterations'].tolist()

    if 'error_feas' in data:
        error_feas = data['error_feas']
        error_opt = data['error_opt']
        visualizer.error_history['error_feas'] = [error_feas[i] for i in range(len(error_feas))]
        visualizer.error_history['error_opt'] = [error_opt[i] for i in range(len(error_opt))]

    # Draw box plot
    result_dir = os.path.dirname(data_path)
    boxplot_path = os.path.join(result_dir, f'{model_type}_error_boxplot_interval.svg')
    visualizer.plot_dual_boxplot_interval(save_path=boxplot_path, interval=1)
    print(f"Error boxplot saved to: {boxplot_path}")


if __name__ == '__main__':
    dscases = {
         'case10ba_ds': TD_case.case10ba_ds(),
        # 'case33bw_ds': TD_case.case33bw_ds(),
        # 'case118zh_ds': TD_case.case118zh_ds(),
        # 'case533mt_hi_ds': TD_case.case533mt_hi_ds(),
        # 'case36real_3phase_ds': DS_case_3phase.case36real_3phase_ds(),
    }
    for casename, ppc in dscases.items():
        is_3phase = '3phase' in casename
        if is_3phase:
            result_dir = f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)\\'
        else:
            result_dir = (f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\'
                     'A(36,2)_type3(1)_lr1(1e-4)_lr2(1e-5)_rate(1e-4)\\')
        for model_type in ['pretrainnet', 'fullnet']:
            data_path = os.path.join(result_dir, f'{model_type}_error_data.npz')

            if os.path.exists(data_path):
                plot_training_error(data_path, casename, model_type)
            else:
                print(f"Data file not found: {data_path}")
                print("Please run first main_ds.py Training model")
