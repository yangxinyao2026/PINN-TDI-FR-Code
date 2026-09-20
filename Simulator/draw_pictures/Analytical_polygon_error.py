# -*- coding: utf-8 -*-
"""Compute analytical-polygon feasibility and optimality errors under parameter perturbations."""

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
# Global configuration
# =============================================================================

N_DTHETA = 100                    # Number of random parameter perturbations
N_CAL = 50                        # Number of error samples per perturbation
DTHETA_RANGE = (-0.5, 0.5)        # Parameter disturbance range


# =============================================================================
# Helper function
# =============================================================================

def vertices_to_halfspace(vertices):
    """Convert ordered convex polygon vertices to Ax <= b half space representation

    for each edge (v_i, v_{i+1}), Compute the outer normal vector asAa line of.
    Args:
        vertices: (m, 2) Sorted array of vertices (arranged counterclockwise)
    Returns:
        A: (m, 2) normal vector matrix
        b: (m,) offset vector
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
    """Get the weight directory and configuration corresponding to the calculation example"""
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
# Data calculation and storage
# =============================================================================

def compute_and_save(casename, ppc, m_directions):
    """Calculates a single configurationanalytical polygonError data and save

    Args:
        casename: Case name
        ppc: Calculation example data
        m_directions: mvalue (8or36)
    """
    is_3phase = '3phase' in casename
    configs = get_case_config(casename)
    config_str = configs[m_directions]
    result_dir = f"{PROJECT_ROOT}/results/ds_proj_paper/{casename}/{config_str}"

    print(f"\n{'='*60}")
    print(f"Calculate: {casename}, m={m_directions}")
    print(f"Results directory: {result_dir}")
    print(f"{'='*60}")

    # Create model
    if is_3phase:
        case_full = DS_case_3phase.DScase_3phase_train(casedata=ppc, model_type='fullnet', device='cpu')
    else:
        case_full = TD_case.DScase_train(casedata=ppc, model_type='fullnet', device='cpu', plot_flag=False)

    model = case_full['errorcalculator'].original_model
    init_params_dict = case_full['params']['params_dict']
    dim_theta = case_full['params']['count']

    # createErrorCalculator
    ec = ErrorCalculator(
        original_model={'model': model},
        A_hat=np.zeros((m_directions, 2)),
        solver='ipopt'
    )
    ec.configure(feas_tol=1e-8, opt_tol=1e-8)

    # Generate randomdtheta
    np.random.seed(42)
    dtheta_list = [np.random.uniform(*DTHETA_RANGE, dim_theta) for _ in range(N_DTHETA)]

    # Store error results
    ap_feas, ap_opt = [], []

    print(f"Start testing {N_DTHETA} random dtheta value...")

    for idx, dtheta in enumerate(dtheta_list):
        print(f"  processing section {idx+1}/{N_DTHETA} a dtheta")

        # Update model parameters
        update_model_parameters(ec, init_params_dict, dtheta=dtheta)

        # Calculateanalytical polygonboundary vertex
        ap_result = compute_analytical_polygon(
            model, init_params_dict, dtheta=None, ppc=ppc,
            error_calculator=ec, n_dirs=m_directions
        )
        vertices = ap_result['boundary_points']
        A_ap, b_ap = vertices_to_halfspace(vertices)

        # Update the approximate polyhedron and calculate the error
        ec.update_polytope(A_hat=A_ap, b_hat=b_ap)
        feas_results, opt_results = ec.calculate(n_cal=N_CAL, cal_feas=True, cal_opt=True)
        feas_errors = [r['error'] for r in feas_results]
        opt_errors = [r['error'] for r in opt_results]
        ap_feas.extend(feas_errors)
        ap_opt.extend(opt_errors)
        print(f"    feas: mean={np.mean(feas_errors):.2e}, max={np.max(feas_errors):.2e} | "
              f"opt: mean={np.mean(opt_errors):.2e}, max={np.max(opt_errors):.2e}")

    # Convert tonumpyarray
    ap_feas = np.array(ap_feas)
    ap_opt = np.array(ap_opt)

    # Calculate statistics
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

    # save data
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

    print(f"\nData has been saved to: {save_path}")
    print(f"  Analytical feas: mean={ap_feas.mean():.2e}, std={ap_feas.std():.2e}, "
          f"median={np.median(ap_feas):.2e}, min={ap_feas.min():.2e}, max={ap_feas.max():.2e}")
    print(f"  Analytical opt:  mean={ap_opt.mean():.2e}, std={ap_opt.std():.2e}, "
          f"median={np.median(ap_opt):.2e}, min={ap_opt.min():.2e}, max={ap_opt.max():.2e}")


# =============================================================================
# main program
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
