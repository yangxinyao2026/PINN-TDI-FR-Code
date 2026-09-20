# -*- coding: utf-8 -*-
"""Evaluate a trained network under random parameter perturbations and save the errors."""

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
# Global configuration
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Error analysis configuration
N_DTHETA = 100                    # Number of random parameter perturbations
N_SAMPLES_PER_DTHETA = 50         # Number of error samples per perturbation
DTHETA_RANGE = (-0.5, 0.5)        # Parameter disturbance range

# Boundary calculation configuration
N_DIRECTIONS = 360                # Calculate the number of directions for boundary points


# =============================================================================
# Helper function
# =============================================================================

def get_weights_dir(casename):
    """Get the weight file directory (single-phase default path) """
    return (f"{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\"
            "A(8,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)")


def load_pretrained_weights(result_dir, A_shape):
    """Load pretrained weights"""
    pretrainnet = PreTrainNet(np.zeros(A_shape), np.zeros(A_shape[0]),
                               is_epigraph=False, device=device)
    weights_path = os.path.join(result_dir, 'pretrainnet_weights.pth')
    pretrainnet.load_state_dict(torch.load(weights_path, map_location=device))
    pretrainnet = pretrainnet.to(device)

    A, b = pretrainnet()
    return A[0].detach().cpu().numpy(), b[0].detach().cpu().numpy()


def compute_original_boundary(model, n_directions, A_hat_shape=(8, 2)):
    """Calculate the boundary points of the original feasible region"""
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
    """Update model load parameters (Pd_meta, Qd_meta), Support single phase/Three phases"""
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
    Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)
    error_calculator.update_parameters({'Pd_meta': Pd_meta_new, 'Qd_meta': Qd_meta_new})


def test_model(fullnet, dtheta, error_calculator):
    """Test a single model on a given dtheta The error below"""
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
# Data calculation and storage
# =============================================================================

def compute_and_save_feasible_region(model, casename, ppc, A_pretrained, b_pretrained,
                                      output_dir, dim_theta, result_dir, A_hat_shape=(8, 2)):
    """Calculate the feasible region comparison data and save it as .npz"""
    dtheta = np.array([0.0] * dim_theta)

    # Calculate the original feasible region boundary
    print("Calculate the original feasible region boundary...")
    boundary_points = compute_original_boundary(model, N_DIRECTIONS, A_hat_shape)

    # initialization FullNet
    fullnet = FullNet(dim_theta=dim_theta, A_init=A_pretrained,
                      b_init=b_pretrained, n_hidden=128, device=device)
    fullnet = fullnet.to(device)

    # moderate Model prediction
    fullnet.load_state_dict(
        torch.load(os.path.join(result_dir, 'fullnet_weights.pth'), map_location=device))
    A_mod, b_mod = fullnet(torch.tensor(dtheta, dtype=torch.float32).to(device))
    A_mod = A_mod[0].detach().cpu().numpy()
    b_mod = b_mod[0].detach().cpu().numpy()

    # feasible Model prediction
    fullnet.load_state_dict(
        torch.load(os.path.join(result_dir, 'fullnet_weights_feasible.pth'), map_location=device))
    A_fea, b_fea = fullnet(torch.tensor(dtheta, dtype=torch.float32).to(device))
    A_fea = A_fea[0].detach().cpu().numpy()
    b_fea = b_fea[0].detach().cpu().numpy()

    # Calculate coordinate range
    baseMVA = ppc['baseMVA']
    bus_P = ppc['bus'][:, 2]
    bus_Q = ppc['bus'][:, 3]
    xlim = np.array([sum(bus_P) - 0.5 * abs(sum(bus_P)),
                     sum(bus_P) + 0.8 * abs(sum(bus_P))]) / baseMVA
    ylim = np.array([sum(bus_Q) - 0.5 * abs(sum(bus_Q)),
                     sum(bus_Q) + 0.8 * abs(sum(bus_Q))]) / baseMVA

    # Save data to the same directory as the image
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'feasible_region_data.npz')
    np.savez(save_path,
             boundary_points=boundary_points,
             A_moderate=A_mod, b_moderate=b_mod,
             A_feasible=A_fea, b_feasible=b_fea,
             xlim=xlim, ylim=ylim,
             dtheta=dtheta, n_directions=N_DIRECTIONS)
    print(f"Feasible region data has been saved to: {save_path}")


def compute_and_save_error_analysis(model, casename, init_params_dict, dim_theta,
                                     A_pretrained, b_pretrained, output_dir, result_dir,
                                     A_hat_shape=(8, 2)):
    """Calculate the error analysis data under random parameter disturbance and save it as .npz"""
    print("\n" + "=" * 60)
    print("Start random dtheta Error analysis")
    print("=" * 60)

    # Create an error calculator
    error_calculator = ErrorCalculator(
        original_model={'model': model},
        A_hat=np.zeros(A_hat_shape),
        solver='ipopt'
    )
    error_calculator.configure(feas_tol=1e-8, opt_tol=1e-8)

    # initialization FullNet
    fullnet = FullNet(dim_theta=dim_theta, A_init=A_pretrained,
                      b_init=b_pretrained, n_hidden=128, device=device)
    fullnet = fullnet.to(device)

    # Generate random dtheta
    np.random.seed(42)
    dtheta_list = [np.random.uniform(*DTHETA_RANGE, dim_theta) for _ in range(N_DTHETA)]

    # Store error results
    moderate_feas, moderate_opt = [], []
    feasible_feas, feasible_opt = [], []

    print(f"Start testing {N_DTHETA} random dtheta value...")

    for idx, dtheta in enumerate(dtheta_list):
        if (idx + 1) % 10 == 0:
            print(f"  processing section {idx + 1}/{N_DTHETA} a dtheta")

        # Update model parameters
        update_model_parameters(error_calculator, dtheta, init_params_dict)

        # test moderate model
        fullnet.load_state_dict(
            torch.load(os.path.join(result_dir, 'fullnet_weights.pth'), map_location=device))
        feas, opt = test_model(fullnet, dtheta, error_calculator)
        moderate_feas.extend(feas)
        moderate_opt.extend(opt)

        # test feasible model
        fullnet.load_state_dict(
            torch.load(os.path.join(result_dir, 'fullnet_weights_feasible.pth'), map_location=device))
        feas, opt = test_model(fullnet, dtheta, error_calculator)
        feasible_feas.extend(feas)
        feasible_opt.extend(opt)

    # Convert to numpy array
    moderate_feas = np.array(moderate_feas)
    moderate_opt = np.array(moderate_opt)
    feasible_feas = np.array(feasible_feas)
    feasible_opt = np.array(feasible_opt)

    # Save data to error_distributions Picture sibling directory
    error_dist_dir = os.path.join(os.path.dirname(output_dir), 'error_distributions')
    os.makedirs(error_dist_dir, exist_ok=True)

    # Compute statistical summary
    stats = compute_statistics(moderate_feas, moderate_opt, feasible_feas, feasible_opt)

    save_path = os.path.join(error_dist_dir, 'error_analysis_data.npz')
    np.savez(save_path,
             moderate_feas=moderate_feas,
             moderate_opt=moderate_opt,
             feasible_feas=feasible_feas,
             feasible_opt=feasible_opt,
             **stats)
    print(f"Error analysis data has been saved to: {save_path}")

    # Output statistical summary to console
    print_statistics(stats)

    print("=" * 60)
    print("Error distribution analysis completed!")
    print("=" * 60)


def compute_statistics(moderate_feas, moderate_opt, feasible_feas, feasible_opt):
    """Compute error statistics summary, returning a dictionary for saving"""
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
    """Output error statistics summary to the console"""
    print("\n" + "=" * 60)
    print("Summary of error statistics:")
    print("=" * 60)

    labels = {
        'moderate_feas': "rate_opt_feas = 0.6 feasibility error",
        'moderate_opt': "rate_opt_feas = 0.6 optimality error",
        'feasible_feas': "rate_opt_feas = 1e-4 feasibility error",
        'feasible_opt': "rate_opt_feas = 1e-4 optimality error",
    }
    fields = ['count', 'mean', 'std', 'min', 'p25', 'median', 'p75', 'max']
    field_names = ['Number of samples', 'mean', 'standard deviation', 'minimum value', '25%Quantile', 'median', '75%Quantile', 'maximum value']

    for prefix, name in labels.items():
        print(f"\n{name}:")
        for field, fname in zip(fields, field_names):
            print(f"  {fname}:   {stats[f'{prefix}_{field}']:.2e}")


# =============================================================================
# main program
# =============================================================================

if __name__ == '__main__':
    # Definition case dictionary
    dscases = {
        # 'case10ba_ds': TD_case.case10ba_ds(),     # For normal operation, open TD_case.py file, modify the174OK
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
        print(f"\nHandle cases: {casename}")
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

        # Create output directory
        output_dir = f'{result_dir}\\figures\\comparison\\feasible\\contrast\\'

        # Load pretrained weights
        A_pretrained, b_pretrained = load_pretrained_weights(result_dir, A_hat_shape)

        # Calculate and save feasible region comparison data
        compute_and_save_feasible_region(model, casename, ppc, A_pretrained, b_pretrained,
                                          output_dir, dim_theta, result_dir,
                                          A_hat_shape=A_hat_shape)

        # Calculate and save error analysis data
        compute_and_save_error_analysis(model, casename, init_params_dict, dim_theta,
                                         A_pretrained, b_pretrained, output_dir, result_dir,
                                         A_hat_shape=A_hat_shape)
