# -*- coding: utf-8 -*-
"""
main_ds_plot.py
功能：读取 main_ds.py 训练过程中保存的误差数据，绘制误差箱线图

使用方法：先运行 main_ds.py 训练模型并保存误差数据，再运行本脚本绘图
"""

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
    """读取训练误差数据并绘制箱线图"""
    data = np.load(data_path)

    # 重建 ErrorVisualizer 的 error_history
    visualizer = ErrorVisualizer()
    visualizer.error_history['iterations'] = data['iterations'].tolist()

    if 'error_feas' in data:
        error_feas = data['error_feas']
        error_opt = data['error_opt']
        visualizer.error_history['error_feas'] = [error_feas[i] for i in range(len(error_feas))]
        visualizer.error_history['error_opt'] = [error_opt[i] for i in range(len(error_opt))]

    # 绘制箱线图
    result_dir = os.path.dirname(data_path)
    boxplot_path = os.path.join(result_dir, f'{model_type}_error_boxplot_interval.svg')
    visualizer.plot_dual_boxplot_interval(save_path=boxplot_path, interval=1)
    print(f"误差箱线图已保存至: {boxplot_path}")


if __name__ == '__main__':
    dscases = {
        # 'case10ba_ds': TD_case.case10ba_ds(),
         'case33bw_ds': TD_case.case33bw_ds(),
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
                     'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)\\')
        for model_type in ['pretrainnet', 'fullnet']:
            data_path = os.path.join(result_dir, f'{model_type}_error_data.npz')

            if os.path.exists(data_path):
                plot_training_error(data_path, casename, model_type)
            else:
                print(f"未找到数据文件: {data_path}")
                print("请先运行 main_ds.py 训练模型")
