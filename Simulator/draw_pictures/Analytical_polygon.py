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
            "A(8,2)_type3(97, 107, 109, 80, 63, 31)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)")


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

    # 更新模型参数
    if dtheta is not None:
        update_model_parameters(error_calculator, init_params_dict, dtheta)

    orig_model = error_calc.original_model

    # 通过求解模型获取可行域内的中心点（保证在 DistFlow 可行域内）
    center_result = error_calc.optimize_direction(np.array([0.0, 0.0]))
    if center_result is not None:
        total_P, total_Q = center_result[0], center_result[1]
        print(f"可行域中心: P={total_P:.4f}, Q={total_Q:.4f} p.u. (来自模型求解)")
    else:
        # 回退：使用负荷总和
        is_3phase = hasattr(model, 'PH')
        if is_3phase:
            total_P = sum(pyo.value(model.Pd[i, ph]) for i in model.BUS for ph in model.PH)
            total_Q = sum(pyo.value(model.Qd[i, ph]) for i in model.BUS for ph in model.PH)
        else:
            total_P = sum(pyo.value(model.Pd[i]) for i in model.BUS)
            total_Q = sum(pyo.value(model.Qd[i]) for i in model.BUS)
        print(f"可行域中心: P={total_P:.4f}, Q={total_Q:.4f} p.u. (来自负荷总和)")
        orig_model.var_proj[0].value = total_P
        orig_model.var_proj[1].value = total_Q

    # 生成 n_dirs 个均匀方向
    theta = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))

    # 沿各方向寻找边界点
    boundary_points = []
    for idx in range(n_dirs):
        if idx % 6 == 0:
            print(f"  方向 {idx+1}/{n_dirs}: θ={np.degrees(theta[idx]):.1f}°")

        # 通用射线约束：d_perp · (x - center) = 0
        # d = (cos θ, sin θ), d_perp = (-sin θ, cos θ)
        sin_t = np.sin(theta[idx])
        cos_t = np.cos(theta[idx])
        constraint_name = f'dir_constraint_{idx}'
        constraint_expr = (-sin_t * (orig_model.var_proj[0] - total_P)
                           + cos_t * (orig_model.var_proj[1] - total_Q) == 0)
        orig_model.add_component(constraint_name, pyo.Constraint(expr=constraint_expr))

        point = error_calc.optimize_direction(directions[idx])
        boundary_points.append(point if point is not None else [total_P, total_Q])

        orig_model.del_component(constraint_name)

    boundary_points = np.array(boundary_points)

    # 按角度排序
    center = np.array([total_P, total_Q])
    angles = [np.arctan2(p[1] - center[1], p[0] - center[0]) for p in boundary_points]
    sorted_points = boundary_points[np.argsort(angles)]

    return {
        'boundary_points': sorted_points,
        'boundary_points_MW': sorted_points * baseMVA,
        'center': np.array([total_P, total_Q])
    }


# =============================================================================
# Original region 计算
# =============================================================================

def calculate_original_boundary(model, n_directions=360):
    """计算原始可行域边界（精度基准）"""
    theta = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))

    error_calc = ErrorCalculator(
        original_model={'model': model},
        A_hat=np.zeros((n_directions, 2)),
        solver='ipopt'
    )

    boundary = np.zeros((n_directions, 2))
    for i in range(n_directions):
        result = error_calc.optimize_direction(directions[i])
        boundary[i] = result if result is not None else boundary[max(0, i-1)]

    return boundary


# =============================================================================
# FullNet 计算
# =============================================================================

def load_fullnet_weights(result_dir, dim_theta, dtheta, A_hat_shape):
    """加载FullNet并计算A、b矩阵"""
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
    print("解析多边形法（m=36）vs NN方法 对比计算")
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
    print(f"耗时: {ap_time:.2f} 秒")

    # 方法2: Original region（仅作精度参考）
    print("\n--- Original region（精度参考）---")
    t0 = time.time()
    original_boundary = calculate_original_boundary(model, n_directions=360)
    orig_time = time.time() - t0
    print(f"耗时: {orig_time:.2f} 秒")

    # 方法3: FullNet
    print("\n--- FullNet ---")
    t0 = time.time()
    dtheta_arr = np.zeros(dim_theta) if dtheta is None else np.asarray(dtheta)
    A_pred, b_pred = load_fullnet_weights(result_dir, dim_theta, dtheta_arr, A_hat_shape)
    fn_time = time.time() - t0
    print(f"耗时: {fn_time:.2f} 秒")

    # 时间统计
    print("\n" + "-" * 40)
    print("计算时间统计:")
    print(f"  解析多边形法(m={N_DIRECTIONS}): {ap_time:.2f} 秒")
    print(f"  original(360方向): {orig_time:.2f} 秒（仅参考）")
    print(f"  fullnet: {fn_time:.2f} 秒")

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

    save_path = os.path.join(comparison_dir, 'analytical_polygon_comparison_data.npz')
    np.savez(save_path,
             polygon_boundary=ap_results['boundary_points'],
             polygon_center=ap_results['center'],
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
    # 'case33bw_ds': TD_case.case33bw_ds(),
    # 'case51ga_ds': TD_case.case51ga_ds(),
    # 'case74_ds': TD_case.case74_ds(),
     'case118zh_ds': TD_case.case118zh_ds(),
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

        dtheta = np.zeros(dim_theta)

        figure_folder = f'{result_dir}\\figures\\comparison\\feasible\\contrast\\'

        compute_and_save_comparison(model, init_params_dict, dim_theta, result_dir,
                                     figure_folder, dtheta=dtheta, ppc=ppc,
                                     A_hat_shape=A_hat_shape)
