# -*- coding: utf-8 -*-
"""
approximate_polygon_coverage.py
功能：计算 Analytical polygon coverage 和 Learned region coverage

指标：从参考运行点 x^ref 沿方向 v 射线，计算各区域的 k_max，
     coverage = k_max_近似 / k_max_原始

主流程：
1. 对每个配置（4算例×2个m值），创建模型和ErrorCalculator
2. 创建射线优化模型，参考点用 pypower runpf 求解
3. 生成N_DTHETA个随机dtheta
4. 对每个dtheta：
   - 更新模型参数，潮流求解得 x^ref
   - compute_analytical_polygon → A_AP, b_AP
   - fullnet_forward → A_NN, b_NN
   - 随机N_DIRS个方向v，计算各区域k_max及coverage
5. 保存coverage数据
"""

import os
import logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase
from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator
from Simulator.draw_pictures.Analytical_polygon_region import (
    compute_analytical_polygon,
    update_model_parameters,
    load_fullnet,
    fullnet_forward,
)

# =============================================================================
# 全局配置
# =============================================================================

N_DTHETA = 50                     # 随机参数扰动数量
N_DIRS = 100                      # 每个扰动的射线方向数
DTHETA_RANGE = (-0.5, 0.5)        # 参数扰动范围


# =============================================================================
# 辅助函数
# =============================================================================

def vertices_to_halfspace(vertices):
    """将有序凸多边形顶点转换为 Ax <= b 半空间表示"""
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
# 参考点计算（pypower 潮流求解，不考虑灵活性）
# =============================================================================

def compute_x_ref_pypower(ppc, init_params_dict, dtheta):
    """用 pypower runpf 求解无灵活性的潮流，返回参考点 x_ref (per unit)"""
    from pypower.api import ppoption, runpf
    import copy

    baseMVA = ppc['baseMVA']
    ppc_copy = copy.deepcopy(ppc)
    ppc_copy['gen'] = ppc_copy['gen'].astype(float)

    # 从 dtheta 计算新的 Pd_meta / Qd_meta
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    dtheta = np.asarray(dtheta)
    Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
    Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)

    # 反归一化：Pd_pu = Pd_meta * 0.4 * Pd_i + 0.8 * Pd_i
    if Pd_init.ndim == 2:
        # 三相：Pd_meta (n_bus, 3)，用 ppc['bus_P']/['bus_Q'] 获取每相负荷
        Pd_i_pu = ppc['bus_P'] / baseMVA   # (n_bus, 3) MW → per unit
        Qd_i_pu = ppc['bus_Q'] / baseMVA
        new_Pd_pu = Pd_meta_new * 0.4 * Pd_i_pu + 0.8 * Pd_i_pu
        new_Qd_pu = Qd_meta_new * 0.4 * Qd_i_pu + 0.8 * Qd_i_pu
        ppc_copy['bus'][:, 2] = new_Pd_pu.sum(axis=1) * baseMVA
        ppc_copy['bus'][:, 3] = new_Qd_pu.sum(axis=1) * baseMVA
    else:
        # 单相：Pd_meta (n_bus,)
        Pd_i_pu = ppc['bus'][:, 2] / baseMVA
        Qd_i_pu = ppc['bus'][:, 3] / baseMVA
        new_Pd_pu = Pd_meta_new * 0.4 * Pd_i_pu + 0.8 * Pd_i_pu
        new_Qd_pu = Qd_meta_new * 0.4 * Qd_i_pu + 0.8 * Qd_i_pu
        ppc_copy['bus'][:, 2] = new_Pd_pu * baseMVA
        ppc_copy['bus'][:, 3] = new_Qd_pu * baseMVA

    # 求解潮流（重定向 stdout+stderr 抑制 pypower 内部输出）
    import sys, io
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        results, success = runpf(ppc_copy, ppopt)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    if success:
        Pg = results['gen'][0, 1]  # MW
        Qg = results['gen'][0, 2]  # MVar
        return np.array([Pg / baseMVA, Qg / baseMVA])
    return None


# =============================================================================
# 射线优化模型（原始可行域 k_max）
# =============================================================================

def create_ray_model(ec):
    """创建射线优化模型：clone original_model，添加射线约束 maximize k"""
    ray_model = ec.original_model.clone()
    dim = len(ray_model.var_proj)

    ray_model.ray_v = pyo.Param(range(dim), mutable=True, initialize=0.0)
    ray_model.ray_xref = pyo.Param(range(dim), mutable=True, initialize=0.0)
    ray_model.k = pyo.Var(bounds=(0, None), initialize=0.0)

    def ray_constraint_rule(m, j):
        return m.var_proj[j] == m.ray_xref[j] + m.k * m.ray_v[j]
    ray_model.ray_con = pyo.Constraint(range(dim), rule=ray_constraint_rule)

    ray_model.min_direction.deactivate()
    ray_model.min_error.deactivate()
    ray_model.max_k = pyo.Objective(expr=ray_model.k, sense=pyo.maximize)

    return ray_model


def update_ray_model_params(ray_model, init_params_dict, dtheta):
    """更新射线模型的 Pd_meta, Qd_meta 参数"""
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    dtheta = np.asarray(dtheta)
    Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
    Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)

    bus_ids = list(ray_model.BUS)
    if Pd_init.ndim == 2:
        # 三相：Pd_meta 按 (bus, phase) 索引
        phase_list = ['a', 'b', 'c']
        for i, bus in enumerate(bus_ids):
            for p, ph in enumerate(phase_list):
                ray_model.Pd_meta[bus, ph] = float(Pd_meta_new[i, p])
                ray_model.Qd_meta[bus, ph] = float(Qd_meta_new[i, p])
    else:
        # 单相：Pd_meta 按 bus 索引
        for i, bus in enumerate(bus_ids):
            ray_model.Pd_meta[bus] = float(Pd_meta_new.flat[i])
            ray_model.Qd_meta[bus] = float(Qd_meta_new.flat[i])


def compute_k_max_ray(ray_model, x_ref, v, solver):
    """求解射线优化模型，返回原始可行域的 k_max"""
    dim = len(ray_model.var_proj)
    for j in range(dim):
        ray_model.ray_xref[j] = float(x_ref[j])
        ray_model.ray_v[j] = float(v[j])
    ray_model.k.value = 0.0

    try:
        result = solver.solve(ray_model)
        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            return ray_model.k.value
    except Exception:
        pass
    return 0.0


# =============================================================================
# 多面体 k_max（解析计算）
# =============================================================================

def compute_k_max_polytope(A, b, x_ref, v):
    """解析计算多面体 Ax<=b 从 x_ref 沿方向 v 的 k_max

    k_max = min { (b_i - A_i·x_ref) / (A_i·v) } for A_i·v > 0
    """
    Av = A @ v
    Ax = A @ x_ref
    k_max = np.inf
    for i in range(len(b)):
        if Av[i] > 1e-12:
            k_i = (b[i] - Ax[i]) / Av[i]
            k_max = min(k_max, k_i)
    return k_max if np.isfinite(k_max) else 0.0


# =============================================================================
# 随机方向生成
# =============================================================================

def generate_random_directions(n_dirs, seed=None):
    """生成 n_dirs 个二维随机单位方向向量"""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, n_dirs)
    return np.column_stack((np.cos(angles), np.sin(angles)))


# =============================================================================
# 数据计算与保存
# =============================================================================

def compute_and_save(casename, ppc, m_directions):
    """计算 Analytical polygon coverage 和 Learned region coverage 并保存"""
    is_3phase = '3phase' in casename
    configs = get_case_config(casename)
    config_str = configs[m_directions]
    result_dir = f"{PROJECT_ROOT}/results/ds_proj_paper/{casename}/{config_str}"

    print(f"\n{'='*60}")
    print(f"Coverage computation: {casename}, m={m_directions}")
    print(f"Result dir: {result_dir}")
    print(f"{'='*60}")

    # 创建模型
    if is_3phase:
        case_full = DS_case_3phase.DScase_3phase_train(casedata=ppc, model_type='fullnet', device='cpu')
    else:
        case_full = TD_case.DScase_train(casedata=ppc, model_type='fullnet', device='cpu', plot_flag=False)

    model = case_full['errorcalculator'].original_model
    init_params_dict = case_full['params']['params_dict']
    dim_theta = case_full['params']['count']
    A_hat_shape = (m_directions, 2)

    # 创建 ErrorCalculator
    ec = ErrorCalculator(
        original_model={'model': model},
        A_hat=np.zeros((m_directions, 2)),
        solver='ipopt'
    )

    # 创建参考点模型（潮流求解，无灵活性）— 使用 pypower runpf
    print("Reference point: pypower runpf (no flexibility)")

    # 创建射线优化模型
    print("Creating ray model...")
    ray_model = create_ray_model(ec)
    ipopt_solver = SolverFactory('ipopt', tee=False)

    # 加载 FullNet
    print("Loading FullNet...")
    fullnet = load_fullnet(result_dir, dim_theta, A_hat_shape)

    # 生成随机 dtheta
    np.random.seed(42)
    dtheta_list = [np.random.uniform(*DTHETA_RANGE, dim_theta) for _ in range(N_DTHETA)]

    # 存储 coverage 结果
    coverage_ap_list, coverage_nn_list = [], []

    print(f"Computing coverage for {N_DTHETA} dtheta x {N_DIRS} directions...")

    for idx, dtheta in enumerate(dtheta_list):
        print(f"  dtheta {idx+1}/{N_DTHETA}")

        # 更新 ErrorCalculator 参数
        update_model_parameters(ec, init_params_dict, dtheta=dtheta)

        # 用 pypower 求解参考点（无灵活性）
        x_ref = compute_x_ref_pypower(ppc, init_params_dict, dtheta)
        if x_ref is None:
            print(f"    Reference point failed, skipping")
            continue
        print(f"    x_ref: P={x_ref[0]:.4f}, Q={x_ref[1]:.4f}")

        # 计算 Analytical polygon 边界
        ap_result = compute_analytical_polygon(
            model, init_params_dict, dtheta=None, ppc=ppc,
            error_calculator=ec, n_dirs=m_directions
        )
        A_ap, b_ap = vertices_to_halfspace(ap_result['boundary_points'])

        # Learned region (FullNet forward)
        A_nn, b_nn = fullnet_forward(fullnet, dtheta)

        # 更新射线模型参数
        update_ray_model_params(ray_model, init_params_dict, dtheta)

        # 生成随机方向
        directions = generate_random_directions(N_DIRS, seed=idx)

        # 计算每个方向的 coverage
        for v in directions:
            # Original region k_max (ipopt)
            k_max_orig = compute_k_max_ray(ray_model, x_ref, v, ipopt_solver)
            if k_max_orig < 1e-12:
                continue

            # Analytical polygon k_max (analytical)
            k_max_ap = compute_k_max_polytope(A_ap, b_ap, x_ref, v)

            # Learned region k_max (analytical)
            k_max_nn = compute_k_max_polytope(A_nn, b_nn, x_ref, v)

            coverage_ap_list.append(k_max_ap / k_max_orig)
            coverage_nn_list.append(k_max_nn / k_max_orig)

        print(f"    Collected {len(coverage_ap_list)} coverage samples")

    # 转换为 numpy 数组
    coverage_ap = np.array(coverage_ap_list)
    coverage_nn = np.array(coverage_nn_list)

    # 统计量
    stats = {}
    for data, prefix in [(coverage_ap, 'coverage_ap'), (coverage_nn, 'coverage_nn')]:
        stats[f'{prefix}_count'] = np.array(len(data))
        stats[f'{prefix}_mean'] = np.array(data.mean())
        stats[f'{prefix}_std'] = np.array(data.std())
        stats[f'{prefix}_min'] = np.array(data.min())
        stats[f'{prefix}_p25'] = np.array(np.percentile(data, 25))
        stats[f'{prefix}_median'] = np.array(np.median(data))
        stats[f'{prefix}_p75'] = np.array(np.percentile(data, 75))
        stats[f'{prefix}_max'] = np.array(data.max())

    # 保存
    save_dir = os.path.join(result_dir, 'figures', 'comparison', 'feasible',
                            'contrast', 'comparison_AnalyticalPolygon')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'coverage_data_m{m_directions}.npz')

    np.savez(save_path,
             coverage_ap=coverage_ap,
             coverage_nn=coverage_nn,
             m_directions=np.array(m_directions),
             n_dtheta=np.array(N_DTHETA),
             n_dirs=np.array(N_DIRS),
             **stats)

    print(f"\nData saved to: {save_path}")
    print(f"  Analytical polygon coverage: mean={coverage_ap.mean():.4f}, median={np.median(coverage_ap):.4f}, "
          f"min={coverage_ap.min():.4f}, max={coverage_ap.max():.4f}")
    print(f"  Learned region coverage:     mean={coverage_nn.mean():.4f}, median={np.median(coverage_nn):.4f}, "
          f"min={coverage_nn.min():.4f}, max={coverage_nn.max():.4f}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    dscases = {
        # 'case33bw_ds': TD_case.case33bw_ds(),
        # 'case118zh_ds': TD_case.case118zh_ds(),
        # 'case533mt_hi_ds': TD_case.case533mt_hi_ds(),
         'case36real_3phase_ds': DS_case_3phase.case36real_3phase_ds(),
    }

    m_values = [36,8]

    for casename, ppc in dscases.items():
        for m in m_values:
            compute_and_save(casename, ppc, m)
