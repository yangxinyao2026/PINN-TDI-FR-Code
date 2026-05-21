# -*- coding: utf-8 -*-
"""
dtheta_region_comparison.py
计算不同参数扰动下真实可行域 vs 两种训练阶段近似多边形的数据，将结果保存为数据文件
绘图请运行 dtheta_region_comparison_plot.py
"""
import os
import logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import numpy as np
import torch
from pyomo.environ import TerminationCondition

from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase
from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, PreTrainNet, FullNet

# ============ 参数配置 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_DIRECTIONS = 360

CASE_NAME = 'case33bw_ds'
#CASE_NAME = 'case118zh_ds'
#CASE_NAME = 'case533mt_hi_ds'
#CASE_NAME = 'case36real_3phase_ds'

CONFIG_STR_3PHASE = 'A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)'
CONFIG_STR_1PHASE = 'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)'


# ============================================================
# 工具函数
# ============================================================

def compute_true_boundary(ec, n_dirs=N_DIRECTIONS):
    """沿多个方向求解原始可行域边界点"""
    theta = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    pts = []
    for d in directions:
        ec.original_model.min_direction.activate()
        ec.original_model.min_error.deactivate()
        for j in range(ec.dim):
            ec.original_model.direction[j] = d[j]
        try:
            res = ec.solver.solve(ec.original_model)
            if res.solver.termination_condition == TerminationCondition.optimal:
                pts.append([ec.original_model.var_proj[j].value for j in range(ec.dim)])
        except:
            pass
    if not pts:
        return np.zeros((1, 2))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


# ============================================================
# 初始化
# ============================================================

def setup():
    """加载案例、模型、网络"""
    print("加载数据与模型...")
    is_3phase = '3phase' in CASE_NAME

    if is_3phase:
        ppc = DS_case_3phase.case36real_3phase_ds()
        case = DS_case_3phase.DScase_3phase_train(casedata=ppc, model_type='fullnet', device=device, plot_flag=False)
        config_str = CONFIG_STR_3PHASE
    else:
        ppc = getattr(TD_case, CASE_NAME)()
        case = TD_case.DScase_train(casedata=ppc, model_type='fullnet', device=device, plot_flag=False)
        config_str = CONFIG_STR_1PHASE

    ec = case['errorcalculator']
    dim_theta = case['params']['count']
    init_params_dict = case['params']['params_dict']

    xlim = np.array([sum(ppc['bus'][:, 2]) - 0.5 * abs(sum(ppc['bus'][:, 2])),
                     sum(ppc['bus'][:, 2]) + 0.8 * abs(sum(ppc['bus'][:, 2]))]) / ppc['baseMVA']
    ylim = np.array([sum(ppc['bus'][:, 3]) - 0.5 * abs(sum(ppc['bus'][:, 3])),
                     sum(ppc['bus'][:, 3]) + 0.8 * abs(sum(ppc['bus'][:, 3]))]) / ppc['baseMVA']

    # 加载预训练权重
    weights_dir = f"{PROJECT_ROOT}/results/ds_proj_paper/{case['casename']}/{config_str}"
    pre = PreTrainNet(case['A_hat'], case['b_hat'], is_epigraph=False, device=device)
    pre.load_state_dict(torch.load(f"{weights_dir}/pretrainnet_weights.pth", map_location=device))
    pre = pre.to(device)
    A0, b0 = pre()
    A0 = A0[0].detach().cpu().numpy()
    b0 = b0[0].detach().cpu().numpy()

    # 创建 FullNet
    net = FullNet(dim_theta=dim_theta, A_init=A0, b_init=b0, n_hidden=128, device=device)
    net = net.to(device)

    info = {
        'ppc': ppc, 'ec': ec, 'net': net,
        'dim_theta': dim_theta, 'init_params_dict': init_params_dict,
        'xlim': xlim, 'ylim': ylim,
        'weights_dir': weights_dir,
    }
    return info


# ============================================================
# 数据计算与保存
# ============================================================

def compute_and_save(info):
    """计算6组场景数据并保存"""
    ec = info['ec']
    net = info['net']
    init_params_dict = info['init_params_dict']
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    total_params = Pd_init.size + Qd_init.size
    xlim, ylim = info['xlim'], info['ylim']
    weights_dir = info['weights_dir']

    scenarios = [
        {'name': '(a)', 'delta':  0.00, 'label': r'$\Delta\theta=0$'},
        {'name': '(b)', 'delta':  0.15, 'label': r'$\Delta\theta=+0.15$'},
        {'name': '(c)', 'delta': -0.15, 'label': r'$\Delta\theta=-0.15$'},
        {'name': '(d)', 'delta':  0.30, 'label': r'$\Delta\theta=+0.30$'},
        {'name': '(e)', 'delta': -0.30, 'label': r'$\Delta\theta=-0.30$'},
        {'name': '(f)', 'delta':  0.45, 'label': r'$\Delta\theta=+0.45$'},
    ]

    save_dict = {
        'xlim': xlim,
        'ylim': ylim,
        'n_directions': N_DIRECTIONS,
    }

    for idx, scen in enumerate(scenarios):
        delta = scen['delta']
        dtheta = np.array([delta] * total_params, dtype=np.float32)

        # 计算真实可行域边界
        ec_copy = ec.copy()
        Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
        Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)
        ec_copy.update_parameters({'Pd_meta': Pd_meta_new, 'Qd_meta': Qd_meta_new})
        true_pts = compute_true_boundary(ec_copy)

        # NN 预测：moderate
        with torch.no_grad():
            dtheta_t = torch.tensor(dtheta, dtype=torch.float32).to(device)
            net.load_state_dict(torch.load(f"{weights_dir}/fullnet_weights.pth", map_location=device))
            net.eval()
            A_mod, b_mod = net(dtheta_t)
        A_mod_np = A_mod[0].detach().cpu().numpy()
        b_mod_np = b_mod[0].detach().cpu().numpy()

        # NN 预测：feasible
        with torch.no_grad():
            net.load_state_dict(torch.load(f"{weights_dir}/fullnet_weights_feasible.pth", map_location=device))
            net.eval()
            A_feas, b_feas = net(dtheta_t)
        A_feas_np = A_feas[0].detach().cpu().numpy()
        b_feas_np = b_feas[0].detach().cpu().numpy()

        # 保存到字典
        key = f'scenario_{idx}'
        save_dict[f'{key}_name'] = scen['name']
        save_dict[f'{key}_label'] = scen['label']
        save_dict[f'{key}_true_pts'] = true_pts
        save_dict[f'{key}_A_mod'] = A_mod_np
        save_dict[f'{key}_b_mod'] = b_mod_np
        save_dict[f'{key}_A_feas'] = A_feas_np
        save_dict[f'{key}_b_feas'] = b_feas_np

        print(f"  {scen['name']} delta={delta:+.2f} done")

    # 保存数据
    out_dir = os.path.join(weights_dir, 'figures', '6_dtheta_region')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'region_comparison_data.npz')
    np.savez(save_path, **save_dict)
    print(f"\n对比数据已保存至: {save_path}")


if __name__ == '__main__':
    info = setup()
    compute_and_save(info)
