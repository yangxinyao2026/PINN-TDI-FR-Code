# -*- coding: utf-8 -*-
"""
Analytical_polygon_error.py
功能：计算 analytical polygon 方法在随机参数扰动下的可行性与最优性误差统计
绘图请运行 Analytical_polygon_error_plot.py

主要流程：
1. 对每个配置（3算例×2个m值），创建模型和ErrorCalculator
2. 生成N_DTHETA个随机dtheta
3. 对每个dtheta：
   - 更新模型参数（Pd_meta, Qd_meta）
   - compute_analytical_polygon() 获取边界顶点
   - vertices_to_halfspace() 转换为 A, b 半空间表示
   - update_polytope(A, b) 更新近似多面体
   - calculate(n_cal=50) 收集feas/opt误差
4. 保存到 comparison_AnalyticalPolygon/analytical_polygon_error_data_m{m}.npz
"""

import os
import logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import numpy as np

from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase
from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator
from Simulator.draw_pictures.Analytical_polygon_region import (
    compute_analytical_polygon,
    update_model_parameters,
)

# =============================================================================
# 全局配置
# =============================================================================

N_DTHETA = 100                    # 随机参数扰动数量
N_CAL = 50                        # 每个扰动的误差采样数
DTHETA_RANGE = (-0.5, 0.5)        # 参数扰动范围


# =============================================================================
# 辅助函数
# =============================================================================

def vertices_to_halfspace(vertices):
    """将有序凸多边形顶点转换为 Ax <= b 半空间表示

    对每条边 (v_i, v_{i+1})，计算外法向量作为A的一行。
    Args:
        vertices: (m, 2) 有序顶点数组（逆时针排列）
    Returns:
        A: (m, 2) 法向量矩阵
        b: (m,) 偏移向量
    """
    n = len(vertices)
    A = np.zeros((n, 2))
    b = np.zeros(n)
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge = v2 - v1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            normal = np.array([0.0, 0.0])
            b[i] = 0.0
        else:
            normal = normal / norm
            b[i] = np.dot(normal, v1)
        A[i] = normal
    return A, b


def get_case_config(casename):
    """获取算例对应的权重目录和配置"""
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


# =============================================================================
# 数据计算与保存
# =============================================================================

def compute_and_save(casename, ppc, m_directions):
    """计算单个配置的analytical polygon误差数据并保存

    Args:
        casename: 算例名称
        ppc: 算例数据
        m_directions: m值（8或36）
    """
    is_3phase = '3phase' in casename
    configs = get_case_config(casename)
    config_str = configs[m_directions]
    result_dir = f"{PROJECT_ROOT}/results/ds_proj_paper/{casename}/{config_str}"

    print(f"\n{'='*60}")
    print(f"计算: {casename}, m={m_directions}")
    print(f"结果目录: {result_dir}")
    print(f"{'='*60}")

    # 创建模型
    if is_3phase:
        case_full = DS_case_3phase.DScase_3phase_train(casedata=ppc, model_type='fullnet', device='cpu')
    else:
        case_full = TD_case.DScase_train(casedata=ppc, model_type='fullnet', device='cpu', plot_flag=False)

    model = case_full['errorcalculator'].original_model
    init_params_dict = case_full['params']['params_dict']
    dim_theta = case_full['params']['count']

    # 创建ErrorCalculator
    ec = ErrorCalculator(
        original_model={'model': model},
        A_hat=np.zeros((m_directions, 2)),
        solver='ipopt'
    )
    ec.configure(feas_tol=1e-8, opt_tol=1e-8)

    # 生成随机dtheta
    np.random.seed(42)
    dtheta_list = [np.random.uniform(*DTHETA_RANGE, dim_theta) for _ in range(N_DTHETA)]

    # 存储误差结果
    ap_feas, ap_opt = [], []

    print(f"开始测试 {N_DTHETA} 个随机 dtheta 值...")

    for idx, dtheta in enumerate(dtheta_list):
        print(f"  处理第 {idx+1}/{N_DTHETA} 个 dtheta")

        # 更新模型参数
        update_model_parameters(ec, init_params_dict, dtheta=dtheta)

        # 计算analytical polygon边界顶点
        ap_result = compute_analytical_polygon(
            model, init_params_dict, dtheta=None, ppc=ppc,
            error_calculator=ec, n_dirs=m_directions
        )
        vertices = ap_result['boundary_points']
        A_ap, b_ap = vertices_to_halfspace(vertices)

        # 更新近似多面体并计算误差
        ec.update_polytope(A_hat=A_ap, b_hat=b_ap)
        feas_results, opt_results = ec.calculate(n_cal=N_CAL, cal_feas=True, cal_opt=True)
        feas_errors = [r['error'] for r in feas_results]
        opt_errors = [r['error'] for r in opt_results]
        ap_feas.extend(feas_errors)
        ap_opt.extend(opt_errors)
        print(f"    feas: mean={np.mean(feas_errors):.2e}, max={np.max(feas_errors):.2e} | "
              f"opt: mean={np.mean(opt_errors):.2e}, max={np.max(opt_errors):.2e}")

    # 转换为numpy数组
    ap_feas = np.array(ap_feas)
    ap_opt = np.array(ap_opt)

    # 计算统计量
    stats = {}
    for errors, prefix in [(ap_feas, 'analytical_feas'), (ap_opt, 'analytical_opt')]:
        stats[f'{prefix}_count'] = np.array(len(errors))
        stats[f'{prefix}_mean'] = np.array(errors.mean())
        stats[f'{prefix}_std'] = np.array(errors.std())
        stats[f'{prefix}_min'] = np.array(errors.min())
        stats[f'{prefix}_p25'] = np.array(np.percentile(errors, 25))
        stats[f'{prefix}_median'] = np.array(np.median(errors))
        stats[f'{prefix}_p75'] = np.array(np.percentile(errors, 75))
        stats[f'{prefix}_max'] = np.array(errors.max())

    # 保存数据
    save_dir = os.path.join(result_dir, 'figures', 'comparison', 'feasible', 'contrast', 'comparison_AnalyticalPolygon')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'analytical_polygon_error_data_m{m_directions}.npz')

    np.savez(save_path,
             analytical_feas=ap_feas,
             analytical_opt=ap_opt,
             m_directions=np.array(m_directions),
             n_dtheta=np.array(N_DTHETA),
             n_cal=np.array(N_CAL),
             **stats)

    print(f"\n数据已保存至: {save_path}")
    print(f"  Analytical feas: mean={ap_feas.mean():.2e}, std={ap_feas.std():.2e}, "
          f"median={np.median(ap_feas):.2e}, min={ap_feas.min():.2e}, max={ap_feas.max():.2e}")
    print(f"  Analytical opt:  mean={ap_opt.mean():.2e}, std={ap_opt.std():.2e}, "
          f"median={np.median(ap_opt):.2e}, min={ap_opt.min():.2e}, max={ap_opt.max():.2e}")


# =============================================================================
# 主程序
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
        for m in m_values:
            compute_and_save(casename, ppc, m)
