# -*- coding: utf-8 -*-
"""
Box_error.py
功能：计算训练好的神经网络在随机参数扰动下的误差表现，将结果保存为数据文件
绘图请运行 figure2merge_error_plot.py

主要流程：
1. 计算原始可行域边界
2. 计算神经网络近似域的 A、b 矩阵
3. 随机参数扰动下的误差分析
4. 将所有数据保存为 .npz 文件（与对应图片保存在同一目录）
"""

import os
import logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import numpy as np
import torch
import pyomo.environ as pyo

from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase
from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, PreTrainNet, FullNet


# =============================================================================
# 全局配置
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 误差分析配置
N_DTHETA = 100                    # 随机参数扰动数量
N_SAMPLES_PER_DTHETA = 50         # 每个扰动的误差采样数
DTHETA_RANGE = (-0.5, 0.5)        # 参数扰动范围

# 边界计算配置
N_DIRECTIONS = 360                # 计算边界点的方向数量


# =============================================================================
# 辅助函数
# =============================================================================

def get_weights_dir(casename):
    """获取权重文件目录（单相默认路径）"""
    return (f"{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\"
            "A(8,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)")


def load_pretrained_weights(result_dir, A_shape):
    """加载预训练权重"""
    pretrainnet = PreTrainNet(np.zeros(A_shape), np.zeros(A_shape[0]),
                               is_epigraph=False, device=device)
    weights_path = os.path.join(result_dir, 'pretrainnet_weights.pth')
    pretrainnet.load_state_dict(torch.load(weights_path, map_location=device))
    pretrainnet = pretrainnet.to(device)

    A, b = pretrainnet()
    return A[0].detach().cpu().numpy(), b[0].detach().cpu().numpy()


def compute_original_boundary(model, n_directions, A_hat_shape=(8, 2)):
    """计算原始可行域的边界点"""
    theta = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))

    error_calc = ErrorCalculator(
        original_model={'model': model},
        A_hat=np.zeros(A_hat_shape),
        solver='ipopt'
    )

    boundary_points = np.zeros((n_directions, 2))
    for i in range(n_directions):
        boundary_points[i] = error_calc.optimize_direction(directions[i])

    return boundary_points


def update_model_parameters(error_calculator, dtheta, init_params_dict):
    """更新模型的负荷参数 (Pd_meta, Qd_meta)，支持单相/三相"""
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
    Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)
    error_calculator.update_parameters({'Pd_meta': Pd_meta_new, 'Qd_meta': Qd_meta_new})


def test_model(fullnet, dtheta, error_calculator):
    """测试单个模型在给定 dtheta 下的误差"""
    A_pred, b_pred = fullnet(torch.tensor(dtheta, dtype=torch.float32).to(device))
    A_pred = A_pred[0].detach().cpu().numpy()
    b_pred = b_pred[0].detach().cpu().numpy()

    error_calculator.update_polytope(A_hat=A_pred, b_hat=b_pred)
    feas_results, opt_results = error_calculator.calculate(
        n_cal=N_SAMPLES_PER_DTHETA, cal_feas=True, cal_opt=True
    )

    feas_errors = [r['error'] for r in feas_results]
    opt_errors = [r['error'] for r in opt_results]

    return feas_errors, opt_errors


# =============================================================================
# 数据计算与保存
# =============================================================================

def compute_and_save_feasible_region(model, casename, ppc, A_pretrained, b_pretrained,
                                      output_dir, dim_theta, result_dir, A_hat_shape=(8, 2)):
    """计算可行域对比数据并保存为 .npz"""
    dtheta = np.array([0.0] * dim_theta)

    # 计算原始可行域边界
    print("计算原始可行域边界...")
    boundary_points = compute_original_boundary(model, N_DIRECTIONS, A_hat_shape)

    # 初始化 FullNet
    fullnet = FullNet(dim_theta=dim_theta, A_init=A_pretrained,
                      b_init=b_pretrained, n_hidden=128, device=device)
    fullnet = fullnet.to(device)

    # moderate 模型预测
    fullnet.load_state_dict(
        torch.load(os.path.join(result_dir, 'fullnet_weights.pth'), map_location=device))
    A_mod, b_mod = fullnet(torch.tensor(dtheta, dtype=torch.float32).to(device))
    A_mod = A_mod[0].detach().cpu().numpy()
    b_mod = b_mod[0].detach().cpu().numpy()

    # feasible 模型预测
    fullnet.load_state_dict(
        torch.load(os.path.join(result_dir, 'fullnet_weights_feasible.pth'), map_location=device))
    A_fea, b_fea = fullnet(torch.tensor(dtheta, dtype=torch.float32).to(device))
    A_fea = A_fea[0].detach().cpu().numpy()
    b_fea = b_fea[0].detach().cpu().numpy()

    # 计算坐标范围
    baseMVA = ppc['baseMVA']
    bus_P = ppc['bus'][:, 2]
    bus_Q = ppc['bus'][:, 3]
    xlim = np.array([sum(bus_P) - 0.5 * abs(sum(bus_P)),
                     sum(bus_P) + 0.8 * abs(sum(bus_P))]) / baseMVA
    ylim = np.array([sum(bus_Q) - 0.5 * abs(sum(bus_Q)),
                     sum(bus_Q) + 0.8 * abs(sum(bus_Q))]) / baseMVA

    # 保存数据到图片同级目录
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'feasible_region_data.npz')
    np.savez(save_path,
             boundary_points=boundary_points,
             A_moderate=A_mod, b_moderate=b_mod,
             A_feasible=A_fea, b_feasible=b_fea,
             xlim=xlim, ylim=ylim,
             dtheta=dtheta, n_directions=N_DIRECTIONS)
    print(f"可行域数据已保存至: {save_path}")


def compute_and_save_error_analysis(model, casename, init_params_dict, dim_theta,
                                     A_pretrained, b_pretrained, output_dir, result_dir,
                                     A_hat_shape=(8, 2)):
    """计算随机参数扰动下的误差分析数据并保存为 .npz"""
    print("\n" + "=" * 60)
    print("开始随机 dtheta 误差分析")
    print("=" * 60)

    # 创建误差计算器
    error_calculator = ErrorCalculator(
        original_model={'model': model},
        A_hat=np.zeros(A_hat_shape),
        solver='ipopt'
    )
    error_calculator.configure(feas_tol=1e-8, opt_tol=1e-8)

    # 初始化 FullNet
    fullnet = FullNet(dim_theta=dim_theta, A_init=A_pretrained,
                      b_init=b_pretrained, n_hidden=128, device=device)
    fullnet = fullnet.to(device)

    # 生成随机 dtheta
    np.random.seed(42)
    dtheta_list = [np.random.uniform(*DTHETA_RANGE, dim_theta) for _ in range(N_DTHETA)]

    # 存储误差结果
    moderate_feas, moderate_opt = [], []
    feasible_feas, feasible_opt = [], []

    print(f"开始测试 {N_DTHETA} 个随机 dtheta 值...")

    for idx, dtheta in enumerate(dtheta_list):
        if (idx + 1) % 10 == 0:
            print(f"  处理第 {idx + 1}/{N_DTHETA} 个 dtheta")

        # 更新模型参数
        update_model_parameters(error_calculator, dtheta, init_params_dict)

        # 测试 moderate 模型
        fullnet.load_state_dict(
            torch.load(os.path.join(result_dir, 'fullnet_weights.pth'), map_location=device))
        feas, opt = test_model(fullnet, dtheta, error_calculator)
        moderate_feas.extend(feas)
        moderate_opt.extend(opt)

        # 测试 feasible 模型
        fullnet.load_state_dict(
            torch.load(os.path.join(result_dir, 'fullnet_weights_feasible.pth'), map_location=device))
        feas, opt = test_model(fullnet, dtheta, error_calculator)
        feasible_feas.extend(feas)
        feasible_opt.extend(opt)

    # 转换为 numpy 数组
    moderate_feas = np.array(moderate_feas)
    moderate_opt = np.array(moderate_opt)
    feasible_feas = np.array(feasible_feas)
    feasible_opt = np.array(feasible_opt)

    # 保存数据到 error_distributions 图片同级目录
    error_dist_dir = os.path.join(os.path.dirname(output_dir), 'error_distributions')
    os.makedirs(error_dist_dir, exist_ok=True)

    # 计算统计摘要
    stats = compute_statistics(moderate_feas, moderate_opt, feasible_feas, feasible_opt)

    save_path = os.path.join(error_dist_dir, 'error_analysis_data.npz')
    np.savez(save_path,
             moderate_feas=moderate_feas,
             moderate_opt=moderate_opt,
             feasible_feas=feasible_feas,
             feasible_opt=feasible_opt,
             **stats)
    print(f"误差分析数据已保存至: {save_path}")

    # 输出统计摘要到控制台
    print_statistics(stats)

    print("=" * 60)
    print("误差分布分析完成！")
    print("=" * 60)


def compute_statistics(moderate_feas, moderate_opt, feasible_feas, feasible_opt):
    """计算误差统计摘要，返回用于保存的字典"""
    stats = {}
    config = [
        (moderate_feas, "moderate_feas"),
        (moderate_opt, "moderate_opt"),
        (feasible_feas, "feasible_feas"),
        (feasible_opt, "feasible_opt"),
    ]
    for errors, prefix in config:
        stats[f'{prefix}_count'] = np.array(len(errors))
        stats[f'{prefix}_mean'] = np.array(errors.mean())
        stats[f'{prefix}_std'] = np.array(errors.std())
        stats[f'{prefix}_min'] = np.array(errors.min())
        stats[f'{prefix}_p25'] = np.array(np.percentile(errors, 25))
        stats[f'{prefix}_median'] = np.array(np.median(errors))
        stats[f'{prefix}_p75'] = np.array(np.percentile(errors, 75))
        stats[f'{prefix}_max'] = np.array(errors.max())
    return stats


def print_statistics(stats):
    """输出误差统计摘要到控制台"""
    print("\n" + "=" * 60)
    print("误差统计摘要:")
    print("=" * 60)

    labels = {
        'moderate_feas': "rate_opt_feas = 0.6 可行性误差",
        'moderate_opt': "rate_opt_feas = 0.6 最优性误差",
        'feasible_feas': "rate_opt_feas = 1e-4 可行性误差",
        'feasible_opt': "rate_opt_feas = 1e-4 最优性误差",
    }
    fields = ['count', 'mean', 'std', 'min', 'p25', 'median', 'p75', 'max']
    field_names = ['样本数', '均值', '标准差', '最小值', '25%分位', '中位数', '75%分位', '最大值']

    for prefix, name in labels.items():
        print(f"\n{name}:")
        for field, fname in zip(fields, field_names):
            print(f"  {fname}:   {stats[f'{prefix}_{field}']:.2e}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    # 定义案例字典
    dscases = {
        # 'case10ba_ds': TD_case.case10ba_ds(),     # 为了能正常运行，打开 TD_case.py 文件，修改第174行
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
        print(f"\n处理案例: {casename}")
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

        # 创建输出目录
        output_dir = f'{result_dir}\\figures\\comparison\\feasible\\contrast\\'

        # 加载预训练权重
        A_pretrained, b_pretrained = load_pretrained_weights(result_dir, A_hat_shape)

        # 计算并保存可行域对比数据
        compute_and_save_feasible_region(model, casename, ppc, A_pretrained, b_pretrained,
                                          output_dir, dim_theta, result_dir,
                                          A_hat_shape=A_hat_shape)

        # 计算并保存误差分析数据
        compute_and_save_error_analysis(model, casename, init_params_dict, dim_theta,
                                         A_pretrained, b_pretrained, output_dir, result_dir,
                                         A_hat_shape=A_hat_shape)
