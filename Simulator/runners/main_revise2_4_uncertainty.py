# -*- coding: utf-8 -*-
"""Run the R2.4 measurement-uncertainty experiment for case33."""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from Simulator import PROJECT_ROOT
from Simulator.Approximator import FullNet, PreTrainNet
from Simulator.cases import TD_case
from Simulator.reference_point import ReferencePointCalculator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================== Default configuration ==============================
CASENAME = 'case33bw_ds'
NOISE_LEVELS = (0.00, 0.01, 0.03, 0.05, 0.10)
TRUE_DTHETA_RANGE = (-0.5, 0.5)

# These defaults reproduce the formal experiment; command-line options can reduce the scale for diagnostics.
N_TRUE_CASES = 50
N_NOISE_REPEATS = 3
N_DIRECTIONS = 10
SEED = 42

ORIGINAL_RESULT_DIR = (
    PROJECT_ROOT / 'results' / 'ds_proj_paper' / CASENAME
    / 'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)'
)
PRETRAIN_WEIGHTS = ORIGINAL_RESULT_DIR / 'pretrainnet_weights.pth'
FULLNET_WEIGHTS = ORIGINAL_RESULT_DIR / 'fullnet_weights_feasible.pth'
OUT = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'uncertainty' / CASENAME
)


def repository_relative_path(path):
    """Return a portable repository-relative metadata path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def parse_noise_levels(text):
    """Put the command line ``0,0.01,0.03`` Convert to non-negative floating point array."""
    levels = np.asarray([float(x.strip()) for x in text.split(',') if x.strip()],
                        dtype=float)
    if levels.size == 0 or np.any(levels < 0):
        raise argparse.ArgumentTypeError('noise-levels Must be a non-negative number, for example 0,0.01,0.03')
    return levels


def update_true_parameters(error_calculator, dtheta_true, init_params_dict):
    """Use only noiseless parameters theta_true update true Pyomo model."""
    pd_init = init_params_dict['Pd_meta']['initial_value']
    qd_init = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd_init.size

    pd_true = pd_init + dtheta_true[:n_pd].reshape(pd_init.shape)
    qd_true = qd_init + dtheta_true[n_pd:].reshape(qd_init.shape)
    error_calculator.update_parameters({
        'Pd_meta': pd_true,
        'Qd_meta': qd_true,
    })


def apply_relative_measurement_noise(dtheta_true, relative_errors,
                                     ppc, init_params_dict):
    """in reality Pd/Qd A multiplicative relative error is applied to the measured value and then converted back to the network input.

    The original manuscript adopts xmin=0.8*x_nom, xmax=1.2*x_nom, Therefore
    x_true = (0.8 + 0.4*meta_true)*x_nom.For parameters with a nominal load of zero, the actual measurement is always
    is zero; its normalized input remains true meta, No artificially injected noise that has no physical meaning.
    """
    pd_init = init_params_dict['Pd_meta']['initial_value']
    qd_init = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd_init.size

    pd_meta_true = pd_init + dtheta_true[:n_pd].reshape(pd_init.shape)
    qd_meta_true = qd_init + dtheta_true[n_pd:].reshape(qd_init.shape)
    eps_pd = relative_errors[:n_pd].reshape(pd_init.shape)
    eps_qd = relative_errors[n_pd:].reshape(qd_init.shape)

    base_mva = ppc['baseMVA']
    pd_nom = np.asarray(ppc['bus'][:, 2] / base_mva).reshape(pd_init.shape)
    qd_nom = np.asarray(ppc['bus'][:, 3] / base_mva).reshape(qd_init.shape)

    pd_true = (0.8 + 0.4 * pd_meta_true) * pd_nom
    qd_true = (0.8 + 0.4 * qd_meta_true) * qd_nom
    pd_measured = pd_true * (1.0 + eps_pd)
    qd_measured = qd_true * (1.0 + eps_qd)

    pd_meta_measured = pd_meta_true.copy()
    qd_meta_measured = qd_meta_true.copy()
    pd_nonzero = np.abs(pd_nom) > 1e-12
    qd_nonzero = np.abs(qd_nom) > 1e-12
    pd_meta_measured[pd_nonzero] = (
        (pd_measured[pd_nonzero] - 0.8 * pd_nom[pd_nonzero])
        / (0.4 * pd_nom[pd_nonzero])
    )
    qd_meta_measured[qd_nonzero] = (
        (qd_measured[qd_nonzero] - 0.8 * qd_nom[qd_nonzero])
        / (0.4 * qd_nom[qd_nonzero])
    )

    measured = np.concatenate([
        (pd_meta_measured - pd_init).reshape(-1),
        (qd_meta_measured - qd_init).reshape(-1),
    ])
    nonzero_load_mask = np.concatenate([
        pd_nonzero.reshape(-1), qd_nonzero.reshape(-1)
    ])
    return measured, nonzero_load_mask


def load_case_and_model(pretrain_path, fullnet_path):
    """Load case33bw_ds and original manuscript main_ds.py Saved baseline model."""
    if CASENAME != 'case33bw_ds':
        raise NotImplementedError('The current script is supported according to the review experiment requirements case33bw_ds.')
    if not pretrain_path.exists():
        raise FileNotFoundError(f'Pretrained weights not found: {pretrain_path}')
    if not fullnet_path.exists():
        raise FileNotFoundError(f'not found FullNet weight: {fullnet_path}')

    ppc = TD_case.case33bw_ds()
    case_full = TD_case.DScase_train(
        casedata=ppc,
        model_type='fullnet',
        plot_flag=False,
        device=device,
    )

    pre_model = PreTrainNet(
        case_full['A_hat'], case_full['b_hat'],
        is_epigraph=False, device=device,
    )
    pre_model.load_state_dict(torch.load(pretrain_path, map_location=device))
    with torch.no_grad():
        a_pre, b_pre = pre_model()
        a_init = a_pre[0].detach().cpu().numpy()
        b_init = b_pre[0].detach().cpu().numpy()

    model = FullNet(
        dim_theta=case_full['params']['count'],
        A_init=a_init,
        b_init=b_init,
        # with original main_ds.py Exactly the same: single hidden layer 128, Default ReLU.
        n_hidden=128,
        device=device,
    ).to(device)
    model.load_state_dict(torch.load(fullnet_path, map_location=device))
    model.eval()
    return ppc, case_full, model


def predict_polytope(model, dtheta_measured):
    """The network only receives parameters with measurement errors theta_measured."""
    with torch.no_grad():
        dt = torch.tensor(dtheta_measured, dtype=torch.float32, device=device)
        a_pred, b_pred = model(dt)
    return (a_pred[0].detach().cpu().numpy(),
            b_pred[0].detach().cpu().numpy())


def calculate_squared_errors(error_calculator, directions):
    """press ErrorCalculator.calculate() The same process calculates the raw squared error in the specified direction.

    The error returned is when the solution is successful and the two boundaries coincide with 0; If a certain step fails to solve the problem, the original code will also be retained.
    of 0, Use at the same time solve_success=False Mark to facilitate the elimination of solution failures during statistics instead of misjudgment as zero error.
    """
    n_dirs = len(directions)
    feas_errors = np.zeros(n_dirs, dtype=float)
    opt_errors = np.zeros(n_dirs, dtype=float)
    feas_success = np.zeros(n_dirs, dtype=bool)
    opt_success = np.zeros(n_dirs, dtype=bool)

    for i, direction in enumerate(directions):
        # Feasibility error: the projected square distance of the predicted polyhedron support point to the real operating domain.
        x_poly = error_calculator.optimize_direction(direction, in_approx=True)
        x_true_proj = error_calculator.project(x_poly) if x_poly is not None else None
        if x_poly is not None and x_true_proj is not None:
            feas_errors[i] = float(np.sum((x_poly - x_true_proj) ** 2))
            feas_success[i] = True

        # Optimality error is the squared distance from the true support point to the predicted polyhedron.
        x_true = error_calculator.optimize_direction(direction, in_approx=False)
        x_poly_proj = error_calculator.project(x_true, to_approx=True) \
            if x_true is not None else None
        if x_true is not None and x_poly_proj is not None:
            opt_errors[i] = float(np.sum((x_true - x_poly_proj) ** 2))
            opt_success[i] = True

    return feas_errors, opt_errors, feas_success, opt_success


def _relative_error_percentages(squared_errors, reference_power_pu_squared):
    """Normalize squared p.u. distance by the true-case root reference power.

    The dimensionless percentage is
    ``100*sqrt(error/(P_ref_pu**2 + Q_ref_pu**2))``.  The true-case axis is
    always the third axis from the end; repeats and directions are the last
    two axes.
    """
    errors = np.asarray(squared_errors, dtype=float)
    reference = np.asarray(reference_power_pu_squared, dtype=float).reshape(-1)
    if errors.ndim < 3 or errors.shape[-3] != reference.size:
        raise ValueError('The true operating condition dimension of the squared error is inconsistent with the number of reference operating points.')
    if np.any(~np.isfinite(reference)) or np.any(reference <= 1e-16):
        raise ValueError('The square unit power of the root node reference operating point must be a positive finite number..')
    denominator_shape = (1,) * (errors.ndim - 3) + (reference.size, 1, 1)
    return 100.0 * np.sqrt(
        np.maximum(errors, 0.0) / reference.reshape(denominator_shape)
    )


def _percentage_values(squared_errors, success, reference_power_pu_squared):
    percentages = _relative_error_percentages(
        squared_errors, reference_power_pu_squared)
    return percentages[success]


def calculate_reference_points(ppc, init_params_dict, true_dthetas,
                               original_model_dict=None):
    """Calculate per-condition root reference points in p.u."""
    calculator = ReferencePointCalculator(
        ppc=ppc,
        init_params_dict=init_params_dict,
        original_model=original_model_dict,
        solver='ipopt',
    )
    points = np.full((len(true_dthetas), 2), np.nan, dtype=float)
    for true_idx, dtheta_true in enumerate(true_dthetas):
        point = calculator.compute(dtheta_true)
        if point is None or not np.all(np.isfinite(point)):
            raise RuntimeError(
                f'Real operating conditions {true_idx} The calculation of the root node reference operating point failed..')
        points[true_idx] = point
    power_squared = np.sum(points ** 2, axis=1)
    if np.any(power_squared <= 1e-16):
        raise RuntimeError('At least one root node reference operating point is close to zero and the relative error cannot be calculated.')
    return points, power_squared


def save_reference_points_csv(path, reference_points_pu, base_mva):
    """Keep the normalization denominator auditable for every true case."""
    rows = []
    for index, point in enumerate(np.asarray(reference_points_pu, dtype=float)):
        squared = float(np.sum(point ** 2))
        rows.append({
            'true_case': index,
            'reference_p_pu': float(point[0]),
            'reference_q_pu': float(point[1]),
            'reference_p_mw': float(point[0] * base_mva),
            'reference_q_mvar': float(point[1] * base_mva),
            'reference_power_pu_squared': squared,
            'reference_apparent_power_pu': float(np.sqrt(squared)),
            'reference_apparent_power_mva': float(np.sqrt(squared) * base_mva),
        })
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_core_summary_csv(path, noise_levels, feas_errors, opt_errors,
                          feas_success, opt_success, feas_pct_increments,
                          opt_pct_increments, paired_feas_success,
                          paired_opt_success, out_of_range_measurement_mask,
                          reference_power_pu_squared):
    """Save the single core table used to answer reviewer comment R2.4."""
    fields = [
        'noise_level',
        'mean_feasibility_error_pct',
        'p95_feasibility_error_pct',
        'paired_feasibility_increment_percentage_points',
        'mean_projection_optimality_error_pct',
        'p95_projection_optimality_error_pct',
        'paired_projection_optimality_increment_percentage_points',
        'out_of_range_measurement_pct',
    ]
    rows = []
    for i, level in enumerate(noise_levels):
        feas_pct = _percentage_values(
            feas_errors[i], feas_success[i], reference_power_pu_squared)
        opt_pct = _percentage_values(
            opt_errors[i], opt_success[i], reference_power_pu_squared)
        paired_feas = feas_pct_increments[i][paired_feas_success[i]]
        paired_opt = opt_pct_increments[i][paired_opt_success[i]]
        rows.append({
            'noise_level': float(level),
            'mean_feasibility_error_pct': (
                float(feas_pct.mean()) if feas_pct.size else np.nan),
            'p95_feasibility_error_pct': (
                float(np.percentile(feas_pct, 95)) if feas_pct.size else np.nan),
            'paired_feasibility_increment_percentage_points': (
                float(paired_feas.mean()) if paired_feas.size else np.nan),
            'mean_projection_optimality_error_pct': (
                float(opt_pct.mean()) if opt_pct.size else np.nan),
            'p95_projection_optimality_error_pct': (
                float(np.percentile(opt_pct, 95)) if opt_pct.size else np.nan),
            'paired_projection_optimality_increment_percentage_points': (
                float(paired_opt.mean()) if paired_opt.size else np.nan),
            'out_of_range_measurement_pct': float(
                100.0 * out_of_range_measurement_mask[i].mean()),
        })

    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_experiment(noise_levels, n_true, n_repeat, n_directions, seed,
                   dtheta_range, pretrain_path, fullnet_path, out_dir):
    zero_indices = np.flatnonzero(np.isclose(noise_levels, 0.0))
    if zero_indices.size != 1:
        raise ValueError(
            'Pairing error increment requirement noise_levels contains exactly one 0% Noise free reference.')
    baseline_idx = int(zero_indices[0])

    ppc, case_full, model = load_case_and_model(pretrain_path, fullnet_path)
    dim_theta = case_full['params']['count']
    init_params_dict = case_full['params']['params_dict']
    error_calculator = case_full['errorcalculator']
    rng = np.random.RandomState(seed)
    true_dthetas = rng.uniform(
        dtheta_range[0], dtheta_range[1], size=(n_true, dim_theta))
    # Calculate one no-flexibility root-bus reference point for every true
    # operating condition.  All noise levels, repeats and directions belonging
    # to that condition share the same denominator in the paired comparison.
    reference_points_pu, reference_power_pu_squared = calculate_reference_points(
        ppc, init_params_dict, true_dthetas,
        original_model_dict=error_calculator.original_model_dict)
    # Same standard noise and same direction in all noise level Multiplexed to form paired comparisons.
    standard_noises = rng.normal(size=(n_true, n_repeat, dim_theta))
    angles = rng.uniform(0.0, 2.0 * np.pi,
                         size=(n_true, n_repeat, n_directions))
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    shape = (len(noise_levels), n_true, n_repeat, n_directions)
    param_shape = (len(noise_levels), n_true, n_repeat, dim_theta)
    feas_errors = np.zeros(shape, dtype=float)
    opt_errors = np.zeros(shape, dtype=float)
    feas_success = np.zeros(shape, dtype=bool)
    opt_success = np.zeros(shape, dtype=bool)
    measured_dthetas = np.zeros(param_shape, dtype=float)
    out_of_range_mask = np.zeros(param_shape, dtype=bool)
    nonzero_load_mask = None

    total = len(noise_levels) * n_true * n_repeat
    done = 0
    t0 = time.time()
    for level_idx, sigma in enumerate(noise_levels):
        print(f'\n[noise={sigma:.1%}] Start testing')
        for true_idx, dtheta_true in enumerate(true_dthetas):
            # Omega always by noiseless theta_true decide; agree true case Just update once.
            update_true_parameters(error_calculator, dtheta_true, init_params_dict)

            for repeat_idx in range(n_repeat):
                relative_errors = sigma * standard_noises[true_idx, repeat_idx]
                measured, current_nonzero_mask = apply_relative_measurement_noise(
                    dtheta_true, relative_errors, ppc, init_params_dict)
                if nonzero_load_mask is None:
                    nonzero_load_mask = current_nonzero_mask
                measured_dthetas[level_idx, true_idx, repeat_idx] = measured
                out_of_range_mask[level_idx, true_idx, repeat_idx] = (
                    (measured < dtheta_range[0]) | (measured > dtheta_range[1]))

                a_pred, b_pred = predict_polytope(model, measured)
                error_calculator.update_polytope(A_hat=a_pred, b_hat=b_pred)

                result = calculate_squared_errors(
                    error_calculator, directions[true_idx, repeat_idx])
                feas_errors[level_idx, true_idx, repeat_idx] = result[0]
                opt_errors[level_idx, true_idx, repeat_idx] = result[1]
                feas_success[level_idx, true_idx, repeat_idx] = result[2]
                opt_success[level_idx, true_idx, repeat_idx] = result[3]

                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    print(f'  Progress {done}/{total} ({done / total:.0%})', flush=True)

    # same true case, repeat, direction One-to-one correspondence at all noise levels.
    # Keep the signed increment: a negative value indicates that the random perturbation happened to reduce the original approximation error.
    feas_increments = feas_errors - feas_errors[baseline_idx][None, ...]
    opt_increments = opt_errors - opt_errors[baseline_idx][None, ...]
    feasibility_error_percentages = _relative_error_percentages(
        feas_errors, reference_power_pu_squared)
    projection_optimality_error_percentages = _relative_error_percentages(
        opt_errors, reference_power_pu_squared)
    feasibility_percentage_increments = (
        feasibility_error_percentages
        - feasibility_error_percentages[baseline_idx][None, ...])
    projection_optimality_percentage_increments = (
        projection_optimality_error_percentages
        - projection_optimality_error_percentages[baseline_idx][None, ...])
    paired_feas_success = (
        feas_success & feas_success[baseline_idx][None, ...])
    paired_opt_success = (
        opt_success & opt_success[baseline_idx][None, ...])

    active_oor = out_of_range_mask[..., nonzero_load_mask]
    out_of_range_measurement_mask = active_oor.any(axis=-1)
    in_range_measurement_mask = ~out_of_range_measurement_mask

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / 'uncertainty_data.npz'
    np.savez_compressed(
        data_path,
        casename=np.array(CASENAME),
        noise_levels=np.asarray(noise_levels),
        true_dtheta_range=np.asarray(dtheta_range),
        true_dthetas=true_dthetas,
        standard_noises=standard_noises,
        relative_measurement_errors=(
            np.asarray(noise_levels)[:, None, None, None]
            * standard_noises[None, ...]
            * nonzero_load_mask[None, None, None, :]
        ),
        nonzero_load_mask=nonzero_load_mask,
        measured_dthetas=measured_dthetas,
        out_of_range_mask=out_of_range_mask,
        directions=directions,
        base_mva=np.array(float(ppc['baseMVA'])),
        reference_points_pu=reference_points_pu,
        reference_points_physical=(
            reference_points_pu * float(ppc['baseMVA'])),
        reference_power_pu_squared=reference_power_pu_squared,
        reference_apparent_power_pu=np.sqrt(reference_power_pu_squared),
        feasibility_errors=feas_errors,
        optimality_errors=opt_errors,
        projection_optimality_errors=opt_errors,
        feasibility_error_increments=feas_increments,
        projection_optimality_error_increments=opt_increments,
        feasibility_error_percentages=feasibility_error_percentages,
        projection_optimality_error_percentages=(
            projection_optimality_error_percentages),
        feasibility_percentage_increments=(
            feasibility_percentage_increments),
        projection_optimality_percentage_increments=(
            projection_optimality_percentage_increments),
        feasibility_solve_success=feas_success,
        optimality_solve_success=opt_success,
        projection_optimality_solve_success=opt_success,
        paired_feasibility_solve_success=paired_feas_success,
        paired_projection_optimality_solve_success=paired_opt_success,
        baseline_noise_index=np.array(baseline_idx),
        out_of_range_measurement_mask=out_of_range_measurement_mask,
        in_range_measurement_mask=in_range_measurement_mask,
        error_definition=np.array('squared_euclidean_distance'),
        percentage_normalization=np.array(
            '100*sqrt(error/(P_ref_pu^2+Q_ref_pu^2))'),
        seed=np.array(seed),
        pretrain_weights=np.array(repository_relative_path(pretrain_path)),
        fullnet_weights=np.array(repository_relative_path(fullnet_path)),
    )

    summary_path = out_dir / 'summary.csv'
    save_reference_points_csv(
        out_dir / 'reference_points.csv', reference_points_pu,
        float(ppc['baseMVA']))
    rows = save_core_summary_csv(
        summary_path, noise_levels, feas_errors, opt_errors,
        feas_success, opt_success, feasibility_percentage_increments,
        projection_optimality_percentage_increments,
        paired_feas_success, paired_opt_success,
        out_of_range_measurement_mask, reference_power_pu_squared)

    print(f'\nTest completed, time consuming {time.time() - t0:.1f}s')
    print(f'original error: {data_path}')
    print(f'Statistical summary: {summary_path}')
    for row in rows:
        print(
            f"  noise={row['noise_level']:.1%} "
            f"feas={row['mean_feasibility_error_pct']:.3f}% "
            f"feas-p95={row['p95_feasibility_error_pct']:.3f}% "
            f"delta-feas={row['paired_feasibility_increment_percentage_points']:.3f}pp "
            f"proj-opt={row['mean_projection_optimality_error_pct']:.3f}% "
            f"proj-opt-p95={row['p95_projection_optimality_error_pct']:.3f}% "
            f"delta-proj-opt={row['paired_projection_optimality_increment_percentage_points']:.3f}pp "
            f"out-of-range={row['out_of_range_measurement_pct']:.1f}% "
        )


def postprocess_existing_results(out_dir):
    """Re-normalize saved raw errors without repeating optimization solves."""
    data_path = out_dir / 'uncertainty_data.npz'
    summary_path = out_dir / 'summary.csv'
    if not data_path.exists():
        raise FileNotFoundError(f'No original error found: {data_path}')

    with np.load(data_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if str(np.asarray(arrays['casename']).item()) != CASENAME:
        raise ValueError('Existing data calculation examples and currentCASENAMEinconsistent.')

    ppc = TD_case.case33bw_ds()
    case_full = TD_case.DScase_train(
        casedata=ppc, model_type='fullnet', plot_flag=False, device=device)
    init_params_dict = case_full['params']['params_dict']
    reference_points_pu, reference_power_pu_squared = calculate_reference_points(
        ppc, init_params_dict, np.asarray(arrays['true_dthetas'], dtype=float),
        original_model_dict=case_full['errorcalculator'].original_model_dict)

    feas_errors = np.asarray(arrays['feasibility_errors'], dtype=float)
    opt_errors = np.asarray(arrays['projection_optimality_errors'], dtype=float)
    feasibility_percentages = _relative_error_percentages(
        feas_errors, reference_power_pu_squared)
    optimality_percentages = _relative_error_percentages(
        opt_errors, reference_power_pu_squared)
    baseline_idx = int(np.asarray(arrays['baseline_noise_index']).item())
    feasibility_increments = (
        feasibility_percentages
        - feasibility_percentages[baseline_idx][None, ...])
    optimality_increments = (
        optimality_percentages
        - optimality_percentages[baseline_idx][None, ...])

    base_mva = float(ppc['baseMVA'])
    arrays.update({
        'base_mva': np.array(base_mva),
        'reference_points_pu': reference_points_pu,
        'reference_points_physical': reference_points_pu * base_mva,
        'reference_power_pu_squared': reference_power_pu_squared,
        'reference_apparent_power_pu': np.sqrt(reference_power_pu_squared),
        'feasibility_error_percentages': feasibility_percentages,
        'projection_optimality_error_percentages': optimality_percentages,
        'feasibility_percentage_increments': feasibility_increments,
        'projection_optimality_percentage_increments': optimality_increments,
        'percentage_normalization': np.array(
            '100*sqrt(error/(P_ref_pu^2+Q_ref_pu^2))'),
    })
    temporary_path = out_dir / 'uncertainty_data.reference_normalized.tmp.npz'
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(data_path)

    rows = save_core_summary_csv(
        summary_path,
        np.asarray(arrays['noise_levels'], dtype=float),
        feas_errors,
        opt_errors,
        np.asarray(arrays['feasibility_solve_success'], dtype=bool),
        np.asarray(arrays['projection_optimality_solve_success'], dtype=bool),
        feasibility_increments,
        optimality_increments,
        np.asarray(arrays['paired_feasibility_solve_success'], dtype=bool),
        np.asarray(arrays['paired_projection_optimality_solve_success'], dtype=bool),
        np.asarray(arrays['out_of_range_measurement_mask'], dtype=bool),
        reference_power_pu_squared,
    )
    save_reference_points_csv(
        out_dir / 'reference_points.csv', reference_points_pu, base_mva)
    print(f'Renormalized to the root node reference operating point: {data_path}')
    print(f'New summary: {summary_path}')
    print(f'Original data backup: {backup_data}')
    for row in rows:
        print(
            f"noise={row['noise_level']:.1%}, "
            f"feas={row['mean_feasibility_error_pct']:.4f}%, "
            f"opt={row['mean_projection_optimality_error_pct']:.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description='R2.4 case33bw_ds Feasibility under measurement error/Optimality squared error distribution')
    parser.add_argument('--noise-levels', type=parse_noise_levels,
                        default=np.asarray(NOISE_LEVELS),
                        help='Pd/Qd Measure the relative Gaussian error standard deviation, such as 0,0.01,0.03,0.05,0.10')
    parser.add_argument('--n-true', type=int, default=N_TRUE_CASES,
                        help='Number of real operating conditions')
    parser.add_argument('--n-repeat', type=int, default=N_NOISE_REPEATS,
                        help='Number of measurement noise repetitions for each real operating condition')
    parser.add_argument('--n-directions', type=int, default=N_DIRECTIONS,
                        help='Number of random directions for each test')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--pretrain-weights', type=Path,
                        default=PRETRAIN_WEIGHTS)
    parser.add_argument('--fullnet-weights', type=Path,
                        default=FULLNET_WEIGHTS)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument(
        '--postprocess-only', action='store_true',
        help='Only use the existing squared error to calculate the normalized percentage of the reference point and do not repeat the optimization experiment..')
    args = parser.parse_args()

    if args.postprocess_only:
        postprocess_existing_results(args.out)
        return

    for name in ('n_true', 'n_repeat', 'n_directions'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be greater than 0')

    print(f'Calculation example: {CASENAME}  Equipment: {device}')
    print(f'Noise level: {args.noise_levels.tolist()}')
    print(f'scale: {args.n_true} a real operating condition × {args.n_repeat} Measurements × '
          f'{args.n_directions} direction')
    print('evaluation relationship: P(theta_measured) vs Omega(theta_true)')
    print('Measurement error: Pd_measured=Pd_true*(1+epsilon), Qd Same reason; do not clip network input')
    print('Original error: the squared Euclidean distance is saved successively; the optimality index is the geometric error based on the projected distance, '
          'Not the regret value of the objective function')
    print('core table: E%=100*sqrt(e/(P_ref_pu^2+Q_ref_pu^2)), '
          'Normalized by the reference operating point of the root node of each real operating condition; '
          'Pairing increment relative to0%Noise, expressed in percentage points')

    run_experiment(
        noise_levels=args.noise_levels,
        n_true=args.n_true,
        n_repeat=args.n_repeat,
        n_directions=args.n_directions,
        seed=args.seed,
        dtheta_range=TRUE_DTHETA_RANGE,
        pretrain_path=args.pretrain_weights,
        fullnet_path=args.fullnet_weights,
        out_dir=args.out,
    )


if __name__ == '__main__':
    main()
