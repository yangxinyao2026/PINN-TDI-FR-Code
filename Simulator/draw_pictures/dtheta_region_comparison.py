# -*- coding: utf-8 -*-
"""Compute feasible-region comparisons under parameter perturbations."""
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

# ============ Parameter configuration ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_DIRECTIONS = 360

CASE_NAME = 'case33bw_ds'
#CASE_NAME = 'case118zh_ds'
#CASE_NAME = 'case533mt_hi_ds'
#CASE_NAME = 'case36real_3phase_ds'

CONFIG_STR_3PHASE = 'A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)'
CONFIG_STR_1PHASE = 'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)'


# ============================================================
# Utility function
# ============================================================

def compute_true_boundary(ec, n_dirs=N_DIRECTIONS):
    """Solve for original feasible region boundary points along multiple directions"""
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
        except Exception:
            pass
    if not pts:
        return np.zeros((1, 2))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


# ============================================================
# initialization
# ============================================================

def setup():
    """Load cases, models, networks"""
    print("Load data and models...")
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

    # Load pretrained weights
    weights_dir = f"{PROJECT_ROOT}/results/ds_proj_paper/{case['casename']}/{config_str}"
    pre = PreTrainNet(case['A_hat'], case['b_hat'], is_epigraph=False, device=device)
    pre.load_state_dict(torch.load(f"{weights_dir}/pretrainnet_weights.pth", map_location=device))
    pre = pre.to(device)
    A0, b0 = pre()
    A0 = A0[0].detach().cpu().numpy()
    b0 = b0[0].detach().cpu().numpy()

    # create FullNet
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
# Data calculation and storage
# ============================================================

def compute_and_save(info):
    """Calculate6Group scene data and save"""
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

        # Calculate the true feasible region boundary
        ec_copy = ec.copy()
        Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
        Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)
        ec_copy.update_parameters({'Pd_meta': Pd_meta_new, 'Qd_meta': Qd_meta_new})
        true_pts = compute_true_boundary(ec_copy)

        # NN Forecast: moderate
        with torch.no_grad():
            dtheta_t = torch.tensor(dtheta, dtype=torch.float32).to(device)
            net.load_state_dict(torch.load(f"{weights_dir}/fullnet_weights.pth", map_location=device))
            net.eval()
            A_mod, b_mod = net(dtheta_t)
        A_mod_np = A_mod[0].detach().cpu().numpy()
        b_mod_np = b_mod[0].detach().cpu().numpy()

        # NN Forecast: feasible
        with torch.no_grad():
            net.load_state_dict(torch.load(f"{weights_dir}/fullnet_weights_feasible.pth", map_location=device))
            net.eval()
            A_feas, b_feas = net(dtheta_t)
        A_feas_np = A_feas[0].detach().cpu().numpy()
        b_feas_np = b_feas[0].detach().cpu().numpy()

        # save to dictionary
        key = f'scenario_{idx}'
        save_dict[f'{key}_name'] = scen['name']
        save_dict[f'{key}_label'] = scen['label']
        save_dict[f'{key}_true_pts'] = true_pts
        save_dict[f'{key}_A_mod'] = A_mod_np
        save_dict[f'{key}_b_mod'] = b_mod_np
        save_dict[f'{key}_A_feas'] = A_feas_np
        save_dict[f'{key}_b_feas'] = b_feas_np

        print(f"  {scen['name']} delta={delta:+.2f} done")

    # save data
    out_dir = os.path.join(weights_dir, 'figures', '6_dtheta_region')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'region_comparison_data.npz')
    np.savez(save_path, **save_dict)
    print(f"\nComparison data has been saved to: {save_path}")


if __name__ == '__main__':
    info = setup()
    compute_and_save(info)
