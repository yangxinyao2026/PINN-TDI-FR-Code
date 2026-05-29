# -*- coding: utf-8 -*-
"""
Analytical_polygon.py
功能：解析多边形法（m=36方向）与NN方法对比的数据计算脚本
将可行域边界计算从8方向扩展为36方向均匀分布
绘图请运行 Analytical_polygon_plot.py

两种方法：
1. 解析多边形法：m个方向的边界点（m=36，与NN方法面数一致，公平对比）
2. FullNet：神经网络近似
"""

import os
import logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import time
import numpy as np
import torch
import pyomo.environ as pyo

from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, PreTrainNet, FullNet
from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 解析多边形法的方向数（与NN方法面数一致）
N_DIRECTIONS = 8


# =============================================================================
# 辅助函数
# =============================================================================

def update_model_parameters(error_calculator, init_params_dict, dtheta=None):
    """更新模型参数，支持单相/三相"""
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    if dtheta is None:
        dtheta = np.zeros(Pd_init.size + Qd_init.size)
    else:
        dtheta = np.asarray(dtheta)
    Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
    Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)
    error_calculator.update_parameters({'Pd_meta': Pd_meta_new, 'Qd_meta': Qd_meta_new})


def get_weights_dir(casename):
    """获取权重文件目录"""
    return (f"{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\"
            "A(8,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)")


# =============================================================================
# 解析多边形法计算（m方向，通用版）
# =============================================================================

def compute_analytical_polygon(model, init_params_dict, dtheta=None, ppc=None,
                               error_calculator=None, n_dirs=N_DIRECTIONS):
    """
    解析多边形法：沿 n_dirs 个均匀方向寻找可行域边界点
    与八边形法原理相同，但方向数可配置，约束使用通用公式
    """
    baseMVA = ppc['baseMVA'] if ppc else 100.0

    # 创建误差计算器（如果未传入）
    if error_calculator is None:
        error_calc = ErrorCalculator(
            original_model={'model': model},
            A_hat=np.zeros((n_dirs, 2)),
            solver='ipopt'
        )
    else:
        error_calc = error_calculator

    # 收紧 ipopt 容差，提高边界点精度
    error_calc.solver.options['tol'] = 1e-11
    error_calc.solver.options['max_iter'] = 10000

    # 更新模型参数
    if dtheta is not None:
        update_model_parameters(error_calculator, init_params_dict, dtheta)

    # 生成 n_dirs 个均匀方向
    theta = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))

    # 直接沿各方向寻找可行域边界点（不使用参考点和垂直约束）
    boundary_points = []
    for idx in range(n_dirs):
        if idx % 6 == 0:
            print(f"  方向 {idx+1}/{n_dirs}: θ={np.degrees(theta[idx]):.1f}°")

        point = error_calc.optimize_direction(directions[idx])
        boundary_points.append(point if point is not None else [0.0, 0.0])

    boundary_points = np.array(boundary_points)

    # 按角度排序（以质心为参考）
    center = boundary_points.mean(axis=0)
    angles = np.arctan2(boundary_points[:, 1] - center[1], boundary_points[:, 0] - center[0])
    sorted_points = boundary_points[np.argsort(angles)]

    return {
        'boundary_points': sorted_points,
        'boundary_points_MW': sorted_points * baseMVA,
    }


# =============================================================================
# Original region 计算
# =============================================================================

def calculate_original_boundary(error_calculator, n_directions=360):
    """计算原始可行域边界（精度基准），复用已更新参数的error_calculator"""
    theta = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))

    boundary = np.zeros((n_directions, 2))
    for i in range(n_directions):
        result = error_calculator.optimize_direction(directions[i])
        boundary[i] = result if result is not None else boundary[max(0, i-1)]

    return boundary


# =============================================================================
# FullNet 计算
# =============================================================================

def load_fullnet(result_dir, dim_theta, A_hat_shape):
    """加载FullNet权重（仅加载一次）"""
    pretrainnet = PreTrainNet(np.zeros(A_hat_shape), np.zeros(A_hat_shape[0]),
                              is_epigraph=False, device=device)
    pretrainnet.load_state_dict(
        torch.load(os.path.join(result_dir, 'pretrainnet_weights.pth'), map_location=device))
    pretrainnet = pretrainnet.to(device)
    A_pre, b_pre = pretrainnet()
    A_pre = A_pre[0].detach().cpu().numpy()
    b_pre = b_pre[0].detach().cpu().numpy()

    fullnet = FullNet(dim_theta=dim_theta, A_init=A_pre, b_init=b_pre,
                      n_hidden=128, device=device)
    fullnet.load_state_dict(
        torch.load(os.path.join(result_dir, 'fullnet_weights_feasible.pth'), map_location=device))
    fullnet = fullnet.to(device)
    fullnet.eval()
    return fullnet


def fullnet_forward(fullnet, dtheta):
    """FullNet前向传播，返回A、b矩阵"""
    with torch.no_grad():
        A, b = fullnet(torch.tensor(dtheta, dtype=torch.float32).to(device))
    return A[0].detach().cpu().numpy(), b[0].detach().cpu().numpy()


# =============================================================================
# 数据计算与保存
# =============================================================================

def compute_and_save_comparison(model, init_params_dict, dim_theta, result_dir,
                                 figure_folder, dtheta=None, ppc=None,
                                 A_hat_shape=(36, 2)):
    """计算解析多边形法与NN方法的数据并保存"""
    print("\n" + "=" * 60)
    print("解析多边形法（m）vs NN方法 对比计算")
    print("=" * 60)

    baseMVA = ppc['baseMVA'] if ppc else 100.0

    # 创建误差计算器
    error_calculator = ErrorCalculator(
        original_model={'model': model},
        A_hat=np.zeros((N_DIRECTIONS, 2)),
        solver='ipopt'
    )

    # 更新模型参数
    update_model_parameters(error_calculator, init_params_dict, dtheta)

    # 方法1: 解析多边形法（m=36）
    print(f"\n--- 解析多边形法（m={N_DIRECTIONS}）---")
    t0 = time.time()
    ap_results = compute_analytical_polygon(model, init_params_dict, dtheta=dtheta,
                                            ppc=ppc, error_calculator=error_calculator,
                                            n_dirs=N_DIRECTIONS)
    ap_time = time.time() - t0
    print(f"耗时: {ap_time:.4f} 秒")

    # 方法2: Original region（仅作精度参考）
    print("\n--- Original region（精度参考）---")
    t0 = time.time()
    original_boundary = calculate_original_boundary(error_calculator, n_directions=360)
    orig_time = time.time() - t0
    print(f"耗时: {orig_time:.4f} 秒")

    # 方法3: FullNet
    print("\n--- FullNet ---")
    print("加载模型权重...")
    fullnet = load_fullnet(result_dir, dim_theta, A_hat_shape)
    dtheta_arr = np.zeros(dim_theta) if dtheta is None else np.asarray(dtheta)
    t0 = time.time()
    A_pred, b_pred = fullnet_forward(fullnet, dtheta_arr)
    fn_time = time.time() - t0
    print(f"前向传播耗时: {fn_time:.4f} 秒")

    # 时间统计
    print("\n" + "-" * 40)
    print("计算时间统计:")
    print(f"  解析多边形法(m={N_DIRECTIONS}): {ap_time:.4f} 秒")
    print(f"  original(360方向): {orig_time:.4f} 秒（仅参考）")
    print(f"  fullnet: {fn_time:.4f} 秒")

    # 计算坐标范围
    bus_P = ppc['bus'][:, 2]
    bus_Q = ppc['bus'][:, 3]
    xlim = np.array([sum(bus_P) - 0.5 * abs(sum(bus_P)),
                     sum(bus_P) + 0.8 * abs(sum(bus_P))]) / baseMVA
    ylim = np.array([sum(bus_Q) - 0.5 * abs(sum(bus_Q)),
                     sum(bus_Q) + 0.8 * abs(sum(bus_Q))]) / baseMVA

    # 保存数据
    comparison_dir = os.path.join(os.path.dirname(figure_folder), 'comparison_AnalyticalPolygon')
    os.makedirs(comparison_dir, exist_ok=True)

    save_path = os.path.join(comparison_dir, 'analytical_polygon_region_comparison_data.npz')
    np.savez(save_path,
             polygon_boundary=ap_results['boundary_points'],
             n_directions=N_DIRECTIONS,
             original_boundary=original_boundary,
             fullnet_A=A_pred,
             fullnet_b=b_pred,
             xlim=xlim,
             ylim=ylim,
             dtheta=np.array(dtheta),
             times=np.array([ap_time, orig_time, fn_time]))
    print(f"\n对比数据已保存至: {save_path}")

    print("=" * 60)
    print("计算完成！")
    print("=" * 60)


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    dscases = {
    # 'case10ba_ds': TD_case.case10ba_ds(),
    # 'case17me_ds': TD_case.case17me_ds(),
     'case33bw_ds': TD_case.case33bw_ds(),
    # 'case51ga_ds': TD_case.case51ga_ds(),
    # 'case74_ds': TD_case.case74_ds(),
    # 'case118zh_ds': TD_case.case118zh_ds(),
    # 'case136ma_ds': TD_case.case136ma_ds(),
    # 'case533mt_hi_ds': TD_case.case533mt_hi_ds(),
    # 'case36real_3phase_ds': DS_case_3phase.case36real_3phase_ds(),
    }

    for casename, ppc in dscases.items():
        print(f'\n处理案例: {casename}')
        is_3phase = '3phase' in casename

        if is_3phase:
            case_full = DS_case_3phase.DScase_3phase_train(casedata=ppc, model_type='fullnet', device=device)
            model = case_full['errorcalculator'].original_model
            init_params_dict = case_full['params']['params_dict']
            dim_theta = case_full['params']['count']
            A_hat_shape = case_full['A_hat'].shape
            result_dir = os.path.dirname(case_full['result_path'])
        else:
            case_full = TD_case.DScase_train(casedata=ppc, model_type='fullnet', device=device, plot_flag=False)
            model = case_full['errorcalculator'].original_model
            init_params_dict = case_full['params']['params_dict']
            dim_theta = case_full['params']['count']
            A_hat_shape = case_full['A_hat'].shape
            result_dir = get_weights_dir(casename)

        dtheta = np.full(dim_theta, 0)

        figure_folder = f'{result_dir}\\figures\\comparison\\feasible\\contrast\\'

        compute_and_save_comparison(model, init_params_dict, dim_theta, result_dir,
                                     figure_folder, dtheta=dtheta, ppc=ppc,
                                     A_hat_shape=A_hat_shape)
