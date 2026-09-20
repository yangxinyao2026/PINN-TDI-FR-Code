# -*- coding: utf-8 -*-
"""Evaluate radial coverage for analytical and learned polygonal regions."""

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
from Simulator.reference_point import (
    REFERENCE_MEMBERSHIP_TOL,
    ReferencePointCalculator,
    reference_point_membership,
)
from Simulator.draw_pictures.Analytical_polygon_region import (
    compute_analytical_polygon,
    update_model_parameters,
    load_fullnet,
    fullnet_forward,
)

# =============================================================================
# Global configuration
# =============================================================================

N_DTHETA = 50                     # Number of random parameter perturbations
N_DIRS = 100                      # Number of ray directions per perturbation
DTHETA_RANGE = (-0.5, 0.5)        # Parameter disturbance range
COVERAGE_FILENAME_SUFFIX = '_2'   # Recalculate after correcting the three-phase reference point; retain the oldcoverageFile


# =============================================================================
# Helper function
# =============================================================================

def vertices_to_halfspace(vertices):
    """Convert ordered convex polygon vertices to Ax <= b half space representation"""
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
# Ray optimization model (original feasible region k_max)
# =============================================================================

def create_ray_model(ec):
    """Create a ray optimization model: clone original_model, Add ray constraints maximize k"""
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
    """Update ray model Pd_meta, Qd_meta parameters"""
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    dtheta = np.asarray(dtheta)
    Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
    Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)

    bus_ids = list(ray_model.BUS)
    if Pd_init.ndim == 2:
        # Three phases: Pd_meta press (bus, phase) Index
        phase_list = ['a', 'b', 'c']
        for i, bus in enumerate(bus_ids):
            for p, ph in enumerate(phase_list):
                ray_model.Pd_meta[bus, ph] = float(Pd_meta_new[i, p])
                ray_model.Qd_meta[bus, ph] = float(Qd_meta_new[i, p])
    else:
        # single phase: Pd_meta press bus Index
        for i, bus in enumerate(bus_ids):
            ray_model.Pd_meta[bus] = float(Pd_meta_new.flat[i])
            ray_model.Qd_meta[bus] = float(Qd_meta_new.flat[i])


def compute_k_max_ray(ray_model, x_ref, v, solver):
    """Solve the ray optimization model and return the original feasible region k_max"""
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
# Polyhedron k_max (Analytical calculation)
# =============================================================================

def compute_k_max_polytope(A, b, x_ref, v):
    """Analytically compute polyhedra Ax<=b from x_ref along direction v of k_max

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
# Random direction generation
# =============================================================================

def generate_random_directions(n_dirs, seed=None):
    """generate n_dirs two-dimensional random unit direction vectors"""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, n_dirs)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def summarize_coverage_ratios(coverage):
    """Define summary coverage, under-coverage, and over-coverage by the radial ratio values of this script.

    for every direction ``rho = k_approx / k_original``:
    coverage = mean(rho), undercoverage = mean(max(1-rho, 0)),
    overcoverage = mean(max(rho-1, 0)).
    """
    coverage = np.asarray(coverage, dtype=float)
    coverage = coverage[np.isfinite(coverage)]
    if coverage.size == 0:
        raise ValueError('No valid radialcoveragesample')
    under = np.maximum(1.0 - coverage, 0.0)
    over = np.maximum(coverage - 1.0, 0.0)
    tol = 1e-8
    return {
        'coverage_percent': 100.0 * float(np.mean(coverage)),
        'undercoverage_percent': 100.0 * float(np.mean(under)),
        'overcoverage_percent': 100.0 * float(np.mean(over)),
        'undercovered_direction_percent': 100.0 * float(np.mean(coverage < 1.0 - tol)),
        'overcovered_direction_percent': 100.0 * float(np.mean(coverage > 1.0 + tol)),
        'coverage_median_percent': 100.0 * float(np.median(coverage)),
        'coverage_p05_percent': 100.0 * float(np.percentile(coverage, 5)),
        'coverage_p95_percent': 100.0 * float(np.percentile(coverage, 95)),
        'coverage_sample_count': int(coverage.size),
    }


# =============================================================================
# Data calculation and storage
# =============================================================================

def compute_and_save(casename, ppc, m_directions):
    """Calculate Analytical polygon coverage and Learned region coverage and save"""
    is_3phase = '3phase' in casename
    configs = get_case_config(casename)
    config_str = configs[m_directions]
    result_dir = f"{PROJECT_ROOT}/results/ds_proj_paper/{casename}/{config_str}"

    print(f"\n{'='*60}")
    print(f"Coverage computation: {casename}, m={m_directions}")
    print(f"Result dir: {result_dir}")
    print(f"{'='*60}")

    # Create model
    if is_3phase:
        case_full = DS_case_3phase.DScase_3phase_train(
            casedata=ppc, model_type='fullnet', plot_flag=False,
            device='cpu')
    else:
        case_full = TD_case.DScase_train(casedata=ppc, model_type='fullnet', device='cpu', plot_flag=False)

    model = case_full['errorcalculator'].original_model
    init_params_dict = case_full['params']['params_dict']
    dim_theta = case_full['params']['count']
    A_hat_shape = (m_directions, 2)

    # create ErrorCalculator
    ec = ErrorCalculator(
        original_model={'model': model},
        A_hat=np.zeros((m_directions, 2)),
        solver='ipopt'
    )

    reference_calculator = ReferencePointCalculator(
        ppc=ppc,
        init_params_dict=init_params_dict,
        original_model=ec.original_model,
        solver='ipopt',
    )

    method = "three-phase Pyomo no-flex model" if is_3phase else "PYPOWER runpf"
    print(f"Reference point: {method} (no flexibility)")

    # Create a ray optimization model
    print("Creating ray model...")
    ray_model = create_ray_model(ec)
    ipopt_solver = SolverFactory('ipopt', tee=False)

    # Load FullNet
    print("Loading FullNet...")
    fullnet = load_fullnet(result_dir, dim_theta, A_hat_shape)

    # Generate random dtheta
    np.random.seed(42)
    dtheta_list = [np.random.uniform(*DTHETA_RANGE, dim_theta) for _ in range(N_DTHETA)]

    # storage coverage result
    coverage_ap_list, coverage_nn_list = [], []

    print(f"Computing coverage for {N_DTHETA} dtheta x {N_DIRS} directions...")

    for idx, dtheta in enumerate(dtheta_list):
        print(f"  dtheta {idx+1}/{N_DTHETA}")

        # update ErrorCalculator parameters
        update_model_parameters(ec, init_params_dict, dtheta=dtheta)

        # Share the same set of no-flexibility reference point definitions with the training code
        x_ref = reference_calculator.compute(dtheta)
        if x_ref is None:
            print(f"    Reference point failed, skipping")
            continue
        print(f"    x_ref: P={x_ref[0]:.4f}, Q={x_ref[1]:.4f}")

        # Calculate Analytical polygon border
        ap_result = compute_analytical_polygon(
            model, init_params_dict, dtheta=None, ppc=ppc,
            error_calculator=ec, n_dirs=m_directions
        )
        A_ap, b_ap = vertices_to_halfspace(ap_result['boundary_points'])

        # Learned region (FullNet forward)
        A_nn, b_nn = fullnet_forward(fullnet, dtheta)

        ap_member, ap_violation = reference_point_membership(
            A_ap, b_ap, x_ref, REFERENCE_MEMBERSHIP_TOL)
        nn_member, nn_violation = reference_point_membership(
            A_nn, b_nn, x_ref, REFERENCE_MEMBERSHIP_TOL)
        if not ap_member:
            print(f"    Analytical polygon skip: x_ref Not in a polyhedron, "
                  f"max violation={ap_violation:.3e}")
        if not nn_member:
            print(f"    Learned region skip: x_ref Not in a polyhedron, "
                  f"max violation={nn_violation:.3e}")

        # Update ray model parameters
        update_ray_model_params(ray_model, init_params_dict, dtheta)

        # Generate random directions
        directions = generate_random_directions(N_DIRS, seed=idx)

        # Calculate for each direction coverage
        for v in directions:
            # Original region k_max (ipopt)
            k_max_orig = compute_k_max_ray(ray_model, x_ref, v, ipopt_solver)
            if k_max_orig < 1e-12:
                continue

            # Analytical polygon k_max (analytical)
            k_max_ap = (compute_k_max_polytope(A_ap, b_ap, x_ref, v)
                        if ap_member else np.nan)

            # Learned region k_max (analytical)
            k_max_nn = (compute_k_max_polytope(A_nn, b_nn, x_ref, v)
                        if nn_member else np.nan)

            if np.isfinite(k_max_ap):
                coverage_ap_list.append(k_max_ap / k_max_orig)
            if np.isfinite(k_max_nn):
                coverage_nn_list.append(k_max_nn / k_max_orig)

        print(f"    Collected {len(coverage_ap_list)} coverage samples")

    # Convert to numpy array
    coverage_ap = np.array(coverage_ap_list)
    coverage_nn = np.array(coverage_nn_list)

    # Statistics
    stats = {}
    for data, prefix in [(coverage_ap, 'coverage_ap'), (coverage_nn, 'coverage_nn')]:
        stats[f'{prefix}_count'] = np.array(len(data))
        if len(data):
            stats[f'{prefix}_mean'] = np.array(data.mean())
            stats[f'{prefix}_std'] = np.array(data.std())
            stats[f'{prefix}_min'] = np.array(data.min())
            stats[f'{prefix}_p25'] = np.array(np.percentile(data, 25))
            stats[f'{prefix}_median'] = np.array(np.median(data))
            stats[f'{prefix}_p75'] = np.array(np.percentile(data, 75))
            stats[f'{prefix}_max'] = np.array(data.max())
        else:
            for suffix in ('mean', 'std', 'min', 'p25', 'median', 'p75', 'max'):
                stats[f'{prefix}_{suffix}'] = np.array(np.nan)

    # save
    save_dir = os.path.join(result_dir, 'figures', 'comparison', 'feasible',
                            'contrast', 'comparison_AnalyticalPolygon')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f'coverage_data_m{m_directions}{COVERAGE_FILENAME_SUFFIX}.npz')

    np.savez(save_path,
             coverage_ap=coverage_ap,
             coverage_nn=coverage_nn,
             m_directions=np.array(m_directions),
             n_dtheta=np.array(N_DTHETA),
             n_dirs=np.array(N_DIRS),
             **stats)

    print(f"\nData saved to: {save_path}")
    print(f"  Analytical polygon coverage: mean={stats['coverage_ap_mean']:.4f}, "
          f"median={stats['coverage_ap_median']:.4f}, min={stats['coverage_ap_min']:.4f}, "
          f"max={stats['coverage_ap_max']:.4f}")
    print(f"  Learned region coverage:     mean={stats['coverage_nn_mean']:.4f}, "
          f"median={stats['coverage_nn_median']:.4f}, min={stats['coverage_nn_min']:.4f}, "
          f"max={stats['coverage_nn_max']:.4f}")


# =============================================================================
# main program
# =============================================================================

if __name__ == '__main__':
    dscases = {
        # 'case33bw_ds': TD_case.case33bw_ds(),
        # 'case118zh_ds': TD_case.case118zh_ds(),
        # 'case533mt_hi_ds': TD_case.case533mt_hi_ds(),
         'case36real_3phase_ds': DS_case_3phase.case36real_3phase_ds(),
    }

    # Recompute only the m = 36 case used in the paper; preserve earlier m = 8 results.
    m_values = [36]

    for casename, ppc in dscases.items():
        for m in m_values:
            compute_and_save(casename, ppc, m)
