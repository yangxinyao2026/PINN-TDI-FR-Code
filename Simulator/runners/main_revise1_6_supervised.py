# -*- coding: utf-8 -*-
"""Run the R1.6 support-point supervised-learning baseline for case33."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from Simulator import PROJECT_ROOT
from Simulator.Approximator import FullNet, PreTrainNet, Trainer, compute_loss
from Simulator.cases import TD_case
from Simulator.reference_point import (
    REFERENCE_MEMBERSHIP_TOL,
    ReferencePointCalculator,
    reference_point_membership,
)
from Simulator.runners.common_case33_test_conditions import (
    CONDITION_PATH as COMMON_TEST_CONDITION_PATH,
    load_common_test_conditions,
)
from Simulator.draw_pictures.approximate_polygon_coverage import (
    compute_k_max_polytope,
    compute_k_max_ray,
    create_ray_model,
    update_ray_model_params,
)


# ---------------------------------------------------------------------------
# Formal experimental configuration (command line parameters can cover the scale)
# ---------------------------------------------------------------------------
CASENAME = 'case33bw_ds'
N_SIDES = 36
LABEL_SIZES = (50, 100, 200, 500, 1000, 2000, 5000)
N_VALIDATION = 50
N_TEST = 50
BATCH_SIZE = 5

# Supervised training has no minimum duration; validation rules control early stopping.
# 10000step only as a safe upper limit.PINNStill maintain the original fixed training scale.
MIN_OPTIMIZER_STEPS = 0
MAX_OPTIMIZER_STEPS = 10000
VALIDATION_INTERVAL = 50
EARLY_STOPPING_PATIENCE = 20
RELATIVE_IMPROVEMENT_TOL = 1e-4
ABSOLUTE_IMPROVEMENT_TOL = 1e-10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
N_ERROR_DIRECTIONS = 10
N_COVERAGE_DIRECTIONS = 100
DTHETA_RANGE = (-0.5, 0.5)
SEED = 42
# PyCharmDirectly after copying to the serverRun: Performing Formal Support Point Supervised Learning Experiments.
RUN_STAGE = 'all'
SUPPORT_LABEL_METHOD = 'support_points_v1_argmax_36_unit_directions'
SUPPORT_CACHE_SCHEMA_VERSION = 3


def support_cache_signature():
    """Returns a complete, stable parameter signature that determines the numerical meaning of the support point label."""
    return json.dumps({
        'schema_version': SUPPORT_CACHE_SCHEMA_VERSION,
        'method': SUPPORT_LABEL_METHOD,
        'n_sides': N_SIDES,
        'direction_normalization': 'unit_euclidean_norm',
        'objective': 'argmax_direction_dot_PQ',
        'solver_call': 'ErrorCalculator.optimize_direction(-direction)',
        'physical_domain': 'original_ac_model',
    }, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


SUPPORT_CACHE_SIGNATURE = support_cache_signature()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = (PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
       / 'supervised' / CASENAME)
SUPPORT_OUT = OUT / 'support_points'
# R1.1Shared with this experiment50a formal test condition/Real domain cache, subject to the current official method directory.
FORMAL_TEST_TRUTH_PATH = SUPPORT_OUT / 'evaluation' / 'test_truth.npz'

EXPERIMENT_OUT = SUPPORT_OUT
LABEL_DIR = SUPPORT_OUT / 'labels'
MODEL_DIR = SUPPORT_OUT / 'models'
EVAL_DIR = SUPPORT_OUT / 'evaluation'

# PINN training is independent of support-point supervision labels.
PINN_DIR = OUT / 'models' / 'pinn_retrained'
PINN_PRETRAIN_WEIGHTS = PINN_DIR / 'pretrainnet_weights.pth'
PINN_FULLNET_WEIGHTS = PINN_DIR / 'fullnet_weights_feasible.pth'
PINN_TRAINING_METRICS = PINN_DIR / 'training_metrics.npz'
UNIFIED_ONLINE_TIMING_CSV = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'learning_methods' / CASENAME
    / 'online_manuscript_protocol_current_hardware'
    / 'online_manuscript_protocol_times.csv')


class BoundaryPointNet(nn.Module):
    """Directly map load cases to fixed-angle numbered36two-dimensional boundary points."""

    def __init__(self, dim_theta, points_init, hidden_size=128, device='cpu'):
        super().__init__()
        points_init = np.asarray(points_init, dtype=np.float32)
        if points_init.shape != (N_SIDES, 2):
            raise ValueError(f'points_initshould be({N_SIDES}, 2), Actually{points_init.shape}')
        self.n_sides = N_SIDES
        self.net = nn.Sequential(
            nn.Linear(dim_theta, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, N_SIDES * 2, device=device),
        )
        # As in FullNet's zero-weight final layer, initialize all label outputs from the first label.
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.as_tensor(
                points_init.reshape(-1), dtype=torch.float32, device=device))

    def forward(self, delta_theta):
        return self.net(delta_theta).reshape(-1, self.n_sides, 2)


def machine_signature():
    """Identifies the server where the timing is located; used to prevent misuse of old machine timings after copying results."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
    return json.dumps({
        'hostname': 'anonymized-host',
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cuda_device': gpu,
        'torch_version': torch.__version__,
    }, ensure_ascii=False, sort_keys=True)


MACHINE_SIGNATURE = machine_signature()


def repository_relative_path(path):
    """Return a portable repository-relative metadata path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def fixed_normals(n_sides=N_SIDES):
    """Same as the original36, unit normal vectors fixedly numbered by angle."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def update_parameters(error_calculator, init_params_dict, dtheta):
    """Use network input dtheta Update real distribution network model."""
    pd0 = init_params_dict['Pd_meta']['initial_value']
    qd0 = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd0.size
    pd = pd0 + np.asarray(dtheta[:n_pd]).reshape(pd0.shape)
    qd = qd0 + np.asarray(dtheta[n_pd:]).reshape(qd0.shape)
    error_calculator.update_parameters({'Pd_meta': pd, 'Qd_meta': qd})


def build_case():
    """Always start from the original case33 Modeling without relying on other reviewer comment scripts."""
    ppc = TD_case.case33bw_ds()
    case = TD_case.DScase_train(
        casedata=ppc, model_type='fullnet', plot_flag=False, device=DEVICE)
    return ppc, case


def _save_support_label_cache(path, dtheta, support_points, attempts,
                              failed_conditions, failed_direction_solves,
                              elapsed, cumulative_times,
                              cumulative_oracle_calls, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dtheta=np.asarray(dtheta, dtype=np.float64),
        support_points=np.asarray(support_points, dtype=np.float64),
        attempts=np.array(attempts),
        failed_conditions=np.array(failed_conditions),
        failed_direction_solves=np.array(failed_direction_solves),
        generation_time=np.array(elapsed),
        cumulative_generation_time=np.asarray(cumulative_times, dtype=float),
        cumulative_oracle_calls=np.asarray(cumulative_oracle_calls, dtype=int),
        seed=np.array(seed),
        machine_signature=np.array(MACHINE_SIGNATURE),
        n_sides=np.array(N_SIDES),
        support_directions=fixed_normals(),
        label_method=np.array(SUPPORT_LABEL_METHOD),
        support_cache_schema_version=np.array(SUPPORT_CACHE_SCHEMA_VERSION),
        support_cache_signature=np.array(SUPPORT_CACHE_SIGNATURE),
        direction_normalization=np.array('unit_euclidean_norm'),
        support_objective=np.array('argmax direction^T [P,Q]'),
        support_solver_call=np.array(
            'ErrorCalculator.optimize_direction(-direction, in_approx=False)'),
        label_definition=np.array(
            '36 ordered support points of the original AC feasible domain; '
            'network target is support_points(theta), not A(theta) or b(theta)'),
        oracle_call_definition=np.array(
            'one original-domain directional support optimization per direction'),
    )


def generate_support_label_split(name, target_count, seed, case, ppc,
                                 overwrite=False, label_dir=None,
                                 fixed_conditions=None):
    """use36Sub-true feasible region support optimization generates a supervised label."""
    output_label_dir = LABEL_DIR if label_dir is None else Path(label_dir)
    path = output_label_dir / f'{name}_labels.npz'
    fixed_conditions = (None if fixed_conditions is None else np.asarray(
        fixed_conditions, dtype=float)[:target_count])
    if fixed_conditions is not None and len(fixed_conditions) != target_count:
        raise ValueError(f'{name}Insufficient public operating conditions: required{target_count}a.')
    use_cache = path.exists() and not overwrite
    if use_cache:
        with np.load(path) as old:
            cached_machine = (str(old['machine_signature'])
                              if 'machine_signature' in old.files else '')
            cached_signature = (str(old['support_cache_signature'])
                                if 'support_cache_signature' in old.files
                                else '')
            cached_theta = np.asarray(old['dtheta'], dtype=float)
        fixed_prefix_ok = (fixed_conditions is None or (
            len(cached_theta) <= len(fixed_conditions)
            and np.allclose(cached_theta, fixed_conditions[:len(cached_theta)],
                            rtol=0.0, atol=1e-12)))
        incompatible_reasons = []
        if cached_machine != MACHINE_SIGNATURE:
            incompatible_reasons.append('Operating environment')
        if cached_signature != SUPPORT_CACHE_SIGNATURE:
            incompatible_reasons.append('support point label signature')
        if not fixed_prefix_ok:
            incompatible_reasons.append('Public operating condition prefix')
        if incompatible_reasons:
            print(f'[{name}/support] Cache incompatibility ('
                  f'{", ".join(incompatible_reasons)}) , will regenerate.')
            use_cache = False

    if use_cache:
        with np.load(path) as old:
            if len(old['dtheta']) >= target_count:
                print(f'[{name}/support] Use caching: {path} ({len(old["dtheta"])}tags) ')
                return path
            dtheta_all = list(np.asarray(old['dtheta'], dtype=float))
            points_all = list(np.asarray(old['support_points'], dtype=float))
            attempts = int(old['attempts'])
            failed_conditions = int(old['failed_conditions'])
            failed_direction_solves = int(old['failed_direction_solves'])
            previous_time = float(old['generation_time'])
            cumulative_times = list(np.asarray(
                old['cumulative_generation_time'], dtype=float))
            cumulative_oracle_calls = list(np.asarray(
                old['cumulative_oracle_calls'], dtype=int))
        print(f'[{name}/support] Continue from cache: existing{len(dtheta_all)}/{target_count}')
    else:
        dtheta_all = []
        points_all = []
        attempts = failed_conditions = 0
        failed_direction_solves = 0
        previous_time = 0.0
        cumulative_times, cumulative_oracle_calls = [], []

    rng = np.random.RandomState(seed)
    if attempts and fixed_conditions is None:
        rng.uniform(*DTHETA_RANGE, size=(attempts, case['params']['count']))

    ec = case['errorcalculator'].copy()
    ec.solver.options['tol'] = 1e-9
    ec.solver.options['max_iter'] = 10000
    directions = fixed_normals()
    oracle_calls = (int(cumulative_oracle_calls[-1])
                    if cumulative_oracle_calls else 0)
    t0 = time.perf_counter()

    while len(dtheta_all) < target_count:
        attempts += 1
        print(
            f'[{name}/support] Start candidate case {attempts}, '
            f'Currently valid={len(dtheta_all)}/{target_count}')
        dtheta = (fixed_conditions[len(dtheta_all)]
                  if fixed_conditions is not None else
                  rng.uniform(*DTHETA_RANGE, size=case['params']['count']))
        update_parameters(ec, case['params']['params_dict'], dtheta)

        points = np.full((N_SIDES, 2), np.nan, dtype=float)
        valid = True
        for i, direction in enumerate(directions):
            # optimize_direction(c)Ask forargmin c^T x; incoming-dJust askargmax d^T x.
            point = ec.optimize_direction(-direction, in_approx=False)
            oracle_calls += 1
            if point is None or not np.all(np.isfinite(point)):
                failed_direction_solves += 1
                valid = False
                break
            points[i] = np.asarray(point, dtype=float).reshape(2)
        if not valid:
            failed_conditions += 1
            if fixed_conditions is not None:
                raise RuntimeError(
                    f'Public operating conditions{name}[{len(dtheta_all)}]Support point solution failed; '
                    'It is prohibited to skip or replace public operating conditions.')
            continue

        dtheta_all.append(dtheta)
        points_all.append(points)
        cumulative_times.append(previous_time + time.perf_counter() - t0)
        cumulative_oracle_calls.append(oracle_calls)

        # Save each valid operating point immediately so interrupted labeling can resume safely.
        elapsed = previous_time + time.perf_counter() - t0
        _save_support_label_cache(
            path, dtheta_all, points_all, attempts, failed_conditions,
            failed_direction_solves,
            elapsed, cumulative_times, cumulative_oracle_calls, seed)
        print(
            f'[{name}/support] {len(dtheta_all)}/{target_count}, '
            f'High level physics calls={oracle_calls}, Failure condition={failed_conditions}, '
            f'Time consuming={elapsed:.1f}s')
    return path


def load_boundary_point_labels(path, count=None):
    """Read support point supervised labels; the only learning goal is36two-dimensional coordinate points."""
    data = np.load(path)
    n = len(data['dtheta']) if count is None else int(count)
    if len(data['dtheta']) < n:
        raise ValueError(f'{path}only{len(data["dtheta"])}tags, required{n}a')
    return (np.asarray(data['dtheta'][:n], dtype=np.float32),
            np.asarray(data['support_points'][:n], dtype=np.float32))


def load_label_dtheta(path, count=None):
    data = np.load(path)
    n = len(data['dtheta']) if count is None else int(count)
    if len(data['dtheta']) < n:
        raise ValueError(f'{path}only{len(data["dtheta"])}tags, required{n}a')
    return np.asarray(data['dtheta'][:n], dtype=np.float32)


def require_same_server_cache(path, description):
    """Caches involved in training timing must be generated by the current server."""
    if not path.exists():
        raise FileNotFoundError(f'not found{description}: {path}')
    with np.load(path) as data:
        cached = (str(data['machine_signature'])
                  if 'machine_signature' in data.files else '')
    if cached != MACHINE_SIGNATURE:
        raise RuntimeError(
            f'{description}from another operating environment: {path}\n'
            'Please regenerate the data on the current server first.')


def supervised_loss(model, theta, points_true):
    """direct supervision36boundary points; the total loss is the average squared Euclidean distance."""
    points_pred = model(theta)
    loss_x = torch.mean((points_pred[..., 0] - points_true[..., 0]) ** 2)
    loss_y = torch.mean((points_pred[..., 1] - points_true[..., 1]) ** 2)
    return loss_x + loss_y, loss_x, loss_y


def evaluate_label_loss(model, loader):
    model.eval()
    totals = np.zeros(3, dtype=float)
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = tuple(x.to(DEVICE) for x in batch)
            values = supervised_loss(model, *batch)
            theta = batch[0]
            bs = len(theta)
            totals += bs * np.array([float(x) for x in values])
            n += bs
    return totals / max(n, 1)


def train_one(label_count, max_steps, learning_rate, case, overwrite=False):
    """Train a label set; the validation set is only used for stopping and model selection, and does not participate in gradients.

    No mandatory minimum number of update steps is set. If verify tagMSEcontinuous
    ``EARLY_STOPPING_PATIENCE`` The inspection did not reach the relative/absolute improvement threshold,
    It is considered that it has entered the verification platform and stopped early.; ``max_steps`` Only as a safe upper limit.
    The final weight always takes the actual lowest verification labelMSE, The physical area indicators are independently audited after the weights are fixed..
    """
    out_dir = MODEL_DIR / f'n{label_count}'
    weights = out_dir / 'supervised_fullnet_weights.pth'
    meta_path = out_dir / 'training_metrics.npz'
    if weights.exists() and meta_path.exists() and not overwrite:
        with np.load(meta_path) as old:
            cached_machine = (str(old['machine_signature'])
                              if 'machine_signature' in old.files else '')
            required_fields = {
                'min_optimizer_steps', 'max_optimizer_steps',
                'validation_interval', 'early_stopping_patience',
                'relative_improvement_tolerance',
                'absolute_improvement_tolerance', 'stopping_reason',
                'best_validation_step', 'converged_by_validation',
                'label_method', 'support_label_cache_method',
                'support_label_cache_signature',
                'prediction_target',
            }
            protocol_ok = required_fields.issubset(old.files)
            if protocol_ok:
                protocol_ok = bool(
                    int(old['min_optimizer_steps'])
                        == min(MIN_OPTIMIZER_STEPS, max_steps)
                    and int(old['max_optimizer_steps']) == max_steps
                    and int(old['validation_interval'])
                        == VALIDATION_INTERVAL
                    and int(old['early_stopping_patience'])
                        == EARLY_STOPPING_PATIENCE
                    and np.isclose(
                        float(old['relative_improvement_tolerance']),
                        RELATIVE_IMPROVEMENT_TOL, rtol=0.0, atol=0.0)
                    and np.isclose(
                        float(old['absolute_improvement_tolerance']),
                        ABSOLUTE_IMPROVEMENT_TOL, rtol=0.0, atol=0.0)
                    and str(old['label_method']) == 'support_points'
                    and str(old['support_label_cache_method'])
                        == SUPPORT_LABEL_METHOD
                    and str(old['support_label_cache_signature'])
                        == SUPPORT_CACHE_SIGNATURE
                    and str(old['prediction_target']) == 'support_points')
        if cached_machine == MACHINE_SIGNATURE and protocol_ok:
            print(f'[train n={label_count}] You already have authority when using this server: {weights}')
            return weights
        reason = ('Another operating environment' if cached_machine != MACHINE_SIGNATURE
                  else 'Old supervised training or early stopping protocols')
        print(f'[train n={label_count}] detected{reason}, will retrain; '
              'Tag cache remains unchanged.')

    train_label_path = LABEL_DIR / 'train_labels.npz'
    validation_label_path = LABEL_DIR / 'validation_labels.npz'
    require_same_server_cache(train_label_path, 'Training labels and generation timing')
    require_same_server_cache(validation_label_path, 'Verify tags and generate timing')
    theta, points = load_boundary_point_labels(train_label_path, label_count)
    theta_val, points_val = load_boundary_point_labels(validation_label_path)
    train_set = TensorDataset(torch.from_numpy(theta), torch.from_numpy(points))
    val_set = TensorDataset(torch.from_numpy(theta_val),
                            torch.from_numpy(points_val))
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              generator=generator, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize all label parameters from the first label to avoid adding initialization variance.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = BoundaryPointNet(
        case['params']['count'], points[0], hidden_size=128,
        device=DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=WEIGHT_DECAY)

    history_step, history_train, history_val = [], [], []
    best_val = np.inf
    best_step = 0
    best_state = None
    significant_reference_val = np.inf
    evaluations_without_improvement = 0
    stopping_reason = 'maximum_steps_reached'
    step = 0
    min_steps = min(MIN_OPTIMIZER_STEPS, max_steps)
    t0 = time.perf_counter()
    while step < max_steps:
        model.train()
        for batch in train_loader:
            batch = tuple(x.to(DEVICE) for x in batch)
            optimizer.zero_grad()
            loss, _, _ = supervised_loss(model, *batch)
            loss.backward()
            optimizer.step()
            step += 1
            if (step == 1 or step % VALIDATION_INTERVAL == 0
                    or step == min_steps or step == max_steps):
                val_loss = evaluate_label_loss(model, val_loader)[0]
                history_step.append(step)
                train_loss_value = loss.detach().item()
                history_train.append(train_loss_value)
                history_val.append(float(val_loss))
                # Select the final weights by the lowest validation-label MSE; the threshold only controls early stopping.
                # patienceWhether to reset to zero to avoid extremely small value fluctuations and extend training indefinitely.
                if val_loss < best_val:
                    best_val = float(val_loss)
                    best_step = int(step)
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                if not np.isfinite(significant_reference_val):
                    significant_improvement = True
                else:
                    required_improvement = max(
                        ABSOLUTE_IMPROVEMENT_TOL,
                        RELATIVE_IMPROVEMENT_TOL
                        * abs(significant_reference_val))
                    significant_improvement = bool(
                        significant_reference_val - val_loss
                        > required_improvement)
                if significant_improvement:
                    significant_reference_val = float(val_loss)
                    evaluations_without_improvement = 0
                else:
                    evaluations_without_improvement += 1
                print(f'[train n={label_count}] step={step}/{max_steps} '
                      f'train={train_loss_value:.3e} val={val_loss:.3e} '
                      f'best={best_val:.3e}@{best_step} '
                      f'patience={evaluations_without_improvement}/'
                      f'{EARLY_STOPPING_PATIENCE}')
                if (evaluations_without_improvement
                        >= EARLY_STOPPING_PATIENCE):
                    stopping_reason = 'validation_plateau'
                    print(
                        f'[train n={label_count}] Verification tagMSEcontinuous'
                        f'{EARLY_STOPPING_PATIENCE}No significant improvement, '
                        f'instep={step}Stop early; beststep={best_step}.')
                    break
            if step >= max_steps:
                break
        if stopping_reason == 'validation_plateau':
            break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate_label_loss(model, val_loader)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    np.savez(
        meta_path,
        label_count=np.array(label_count),
        optimizer_steps=np.array(step),
        min_optimizer_steps=np.array(min_steps),
        max_optimizer_steps=np.array(max_steps),
        batch_size=np.array(BATCH_SIZE),
        learning_rate=np.array(learning_rate),
        label_method=np.array('support_points'),
        support_label_cache_method=np.array(SUPPORT_LABEL_METHOD),
        support_label_cache_signature=np.array(SUPPORT_CACHE_SIGNATURE),
        prediction_target=np.array('support_points'),
        machine_signature=np.array(MACHINE_SIGNATURE),
        train_time=np.array(train_time),
        best_validation_loss=np.array(best_val),
        best_validation_step=np.array(best_step),
        validation_interval=np.array(VALIDATION_INTERVAL),
        early_stopping_patience=np.array(EARLY_STOPPING_PATIENCE),
        relative_improvement_tolerance=np.array(RELATIVE_IMPROVEMENT_TOL),
        absolute_improvement_tolerance=np.array(ABSOLUTE_IMPROVEMENT_TOL),
        evaluations_without_improvement=np.array(
            evaluations_without_improvement),
        stopping_reason=np.array(stopping_reason),
        converged_by_validation=np.array(
            stopping_reason == 'validation_plateau'),
        checkpoint_selection_metric=np.array(
            'validation_label_total_mse'),
        physical_audit_used_for_selection=np.array(False),
        validation_total_mse=np.array(final_val[0]),
        validation_boundary_x_mse=np.array(final_val[1]),
        validation_boundary_y_mse=np.array(final_val[2]),
        validation_boundary_point_mse=np.array(final_val[0]),
        history_step=np.asarray(history_step),
        history_train_loss=np.asarray(history_train),
        history_validation_loss=np.asarray(history_val),
    )
    print(f'[train n={label_count}] Complete: steps={step}, '
          f'stop={stopping_reason}, best={best_val:.3e}@{best_step}, '
          f'{train_time:.1f}s, weight={weights}')
    return weights


def _prefixed_probe_statistics(prefix, trainer):
    """put Trainer The actual count written can be saved directly to npz scalar field."""
    return {f'{prefix}_{key}': np.array(int(value))
            for key, value in trainer.probe_statistics.items()}


def train_pinn_same_server(ppc, case_full, overwrite=False, quick=False):
    """Reproduce from scratch on current server case33 The pre-training and two-stage physics-informed training.

    Do not read ``results/ds_proj_paper`` weights or timings, and does not write new results back to the original directory.
    Learning rate and stage size adoption case33 Original configuration: pre-training 1e-1/P_rated, FullNet first stage
    3e-5/P_rated, second stage 1e-5/P_rated; The direction numbers are respectively5and2.
    """
    if (PINN_PRETRAIN_WEIGHTS.exists() and PINN_FULLNET_WEIGHTS.exists()
            and PINN_TRAINING_METRICS.exists() and not overwrite):
        with np.load(PINN_TRAINING_METRICS) as old:
            old_quick = bool(old['quick']) if 'quick' in old.files else False
            cached_machine = (str(old['machine_signature'])
                              if 'machine_signature' in old.files else '')
        # Formal results are available forquickCheck; quickWeights must never pass for official results.
        if cached_machine == MACHINE_SIGNATURE and (quick or not old_quick):
            print(f'[PINN] Use the existing ones on this serverR1.6Result of retraining: {PINN_DIR}')
            return PINN_FULLNET_WEIGHTS
        print('[PINN] Another server timing is detected orquickweights, will be automatically retrained.')

    PINN_DIR.mkdir(parents=True, exist_ok=True)
    p_rated = float(np.sum(ppc['bus'][:, 2]) / ppc['baseMVA'])
    if p_rated <= 0:
        raise ValueError(f'Invalid P_rated={p_rated}')
    pipeline_start = time.perf_counter()

    # ---------------- PreTrainNet: Train from scratch without loading any original weights ----------------
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    case_pre = TD_case.DScase_train(
        casedata=ppc, model_type='pretrainnet', plot_flag=False, device=DEVICE)
    case_pre['trainer_configure']['training_callback'] = None
    pre_model = PreTrainNet(
        case_pre['A_hat'], case_pre['b_hat'], is_epigraph=False,
        device=DEVICE).to(DEVICE)
    pre_trainer = Trainer(
        model=pre_model, error_calculator=case_pre['errorcalculator'],
        compute_loss=compute_loss)
    pre_trainer.configure(**case_pre['trainer_configure'])
    pre_trainer.configure(lr=1e-1/p_rated, rate_opt_feas=0.6)
    pre_trainer.initialize()
    t0 = time.perf_counter()
    # Trainerinternalrange(n_train+1), pass2000actual execution2001Updated, consistent with the original.
    pretrain_arg = 5 if quick else 500*4
    phase1_arg = 1 if quick else 20*4-1
    phase2_arg = 0 if quick else 20*2
    pre_trainer.train(n_train=pretrain_arg, params_data=case_pre['params'],
                      parallel=False)
    pretrain_time = time.perf_counter() - t0
    torch.save(pre_model.state_dict(), PINN_PRETRAIN_WEIGHTS)
    print(f'[PINN pretrain] Complete: {pretrain_time:.1f}s, '
          f'steps={pre_trainer.probe_statistics["optimizer_steps"]}')

    # ---------------- FullNet: Start with the pre-trained parameters you just generated ----------------
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    case_full['trainer_configure']['training_callback'] = None
    with torch.no_grad():
        a_pre, b_pre = pre_model()
        a_init = a_pre[0].detach().cpu().numpy()
        b_init = b_pre[0].detach().cpu().numpy()
    full_model = FullNet(
        dim_theta=case_full['params']['count'], A_init=a_init, b_init=b_init,
        hidden_sizes=[128], activation='relu', device=DEVICE).to(DEVICE)
    full_trainer = Trainer(
        model=full_model, error_calculator=case_full['errorcalculator'],
        compute_loss=compute_loss)
    full_trainer.configure(**case_full['trainer_configure'])

    # The first stage: comprehensive feasibility and projection optimality.
    full_trainer.configure(lr=3e-5/p_rated, rate_opt_feas=0.6)
    full_trainer.initialize()
    t0 = time.perf_counter()
    full_trainer.train(n_train=phase1_arg, params_data=case_full['params'],
                       parallel=False)
    phase1_time = time.perf_counter() - t0

    # Phase 2 lowers the learning rate and emphasizes feasibility; initialize() preserves call counts.
    full_trainer.configure(lr=1e-5/p_rated, rate_opt_feas=1e-4)
    full_trainer.initialize()
    t0 = time.perf_counter()
    full_trainer.train(n_train=phase2_arg, params_data=case_full['params'],
                       parallel=False)
    phase2_time = time.perf_counter() - t0
    torch.save(full_model.state_dict(), PINN_FULLNET_WEIGHTS)

    fullnet_time = phase1_time + phase2_time
    pipeline_time = time.perf_counter() - pipeline_start
    np.savez(
        PINN_TRAINING_METRICS,
        quick=np.array(bool(quick)),
        device=np.array(str(DEVICE)),
        machine_signature=np.array(MACHINE_SIGNATURE),
        p_rated=np.array(p_rated),
        pretrain_lr=np.array(1e-1/p_rated),
        phase1_lr=np.array(3e-5/p_rated),
        phase2_lr=np.array(1e-5/p_rated),
        pretrain_time=np.array(pretrain_time),
        phase1_time=np.array(phase1_time),
        phase2_time=np.array(phase2_time),
        fullnet_time=np.array(fullnet_time),
        total_train_time=np.array(pretrain_time + fullnet_time),
        pipeline_wall_time=np.array(pipeline_time),
        **_prefixed_probe_statistics('pretrain', pre_trainer),
        **_prefixed_probe_statistics('fullnet', full_trainer),
    )
    print(f'[PINN fullnet] phase1={phase1_time:.1f}s, phase2={phase2_time:.1f}s, '
          f'steps={full_trainer.probe_statistics["optimizer_steps"]}')
    print(f'[PINN] Total training time on the same server={pretrain_time+fullnet_time:.1f}s, '
          f'result={PINN_DIR}')
    return PINN_FULLNET_WEIGHTS


def load_supervised_model(case, label_count):
    _, points = load_boundary_point_labels(LABEL_DIR / 'train_labels.npz', 1)
    model = BoundaryPointNet(
        case['params']['count'], points[0], hidden_size=128,
        device=DEVICE).to(DEVICE)
    path = MODEL_DIR / f'n{label_count}' / 'supervised_fullnet_weights.pth'
    meta_path = MODEL_DIR / f'n{label_count}' / 'training_metrics.npz'
    if not meta_path.exists():
        raise FileNotFoundError(f'Supervised training metadata not found: {meta_path}')
    with np.load(meta_path) as meta:
        compatible = bool(
            'label_method' in meta.files
            and str(meta['label_method']) == 'support_points'
            and 'support_label_cache_signature' in meta.files
            and str(meta['support_label_cache_signature'])
                == SUPPORT_CACHE_SIGNATURE
            and 'prediction_target' in meta.files
            and str(meta['prediction_target']) == 'support_points')
    if not compatible:
        raise RuntimeError(
            f'n={label_count}The weights are not trained by the current support point labeling protocol.: {path}\n'
            'Please run first --stage train (or default pilot phase) retraining.')
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def load_pinn_model(case):
    """Only load thisR1.6The script is retrained on the current serverPINN."""
    if not PINN_PRETRAIN_WEIGHTS.exists() or not PINN_FULLNET_WEIGHTS.exists():
        raise FileNotFoundError(
            f'not foundR1.6RetrainPINN: {PINN_DIR}\n'
            'Please run first --stage pinn; Will not fall back on reading original weights.')
    require_same_server_cache(PINN_TRAINING_METRICS, 'R1.6 PINNTraining timing')
    pre = PreTrainNet(case['A_hat'], case['b_hat'], is_epigraph=False,
                      device=DEVICE)
    pre.load_state_dict(torch.load(PINN_PRETRAIN_WEIGHTS, map_location=DEVICE))
    with torch.no_grad():
        a0, b0 = pre()
    model = FullNet(case['params']['count'], a0[0].cpu().numpy(),
                    b0[0].cpu().numpy(), n_hidden=128, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PINN_FULLNET_WEIGHTS, map_location=DEVICE))
    model.eval()
    return model


def predict(model, dtheta):
    with torch.no_grad():
        x = torch.as_tensor(dtheta, dtype=torch.float32, device=DEVICE)
        output = model(x)
        if isinstance(model, BoundaryPointNet):
            return output.detach().cpu().numpy()
        a, b = output
        return a.detach().cpu().numpy(), b.detach().cpu().numpy()


def load_unified_online_times(required=False):
    """Read the unified original process retest results; R1.6The review itself no longer carries online timing."""
    if not UNIFIED_ONLINE_TIMING_CSV.exists():
        if required:
            raise FileNotFoundError(
                'Unified online timing results are missing, please run: \n'
                'Simulator/runners/main_revise1_1_online_cold_timing.py\n'
                f'expected file: {UNIFIED_ONLINE_TIMING_CSV}')
        return {}
    with UNIFIED_ONLINE_TIMING_CSV.open(
            'r', encoding='utf-8-sig', newline='') as handle:
        return {
            row['method']: float(row['online_seconds'])
            for row in csv.DictReader(handle)
        }


def prepare_truth(case, ppc, n_test, n_error_dirs, n_coverage_dirs,
                  overwrite=False, test_label_path=None, truth_path=None,
                  theta_override=None):
    """Test conditions common to all methods/The true feasible region information is only calculated once in the direction."""
    path = (EVAL_DIR / 'test_truth.npz'
            if truth_path is None else Path(truth_path))
    label_path = (LABEL_DIR / 'test_labels.npz'
                  if test_label_path is None else Path(test_label_path))
    theta_test = (load_label_dtheta(label_path, n_test)
                  if theta_override is None else
                  np.asarray(theta_override, dtype=np.float32)[:n_test])
    if theta_test.shape != (n_test, case['params']['count']):
        raise ValueError(
            f'The shape of the test condition should be({n_test}, {case["params"]["count"]}), '
            f'Actually{theta_test.shape}.')
    if path.exists() and not overwrite:
        with np.load(path) as data:
            shape_ok = (len(data['dtheta']) == n_test
                        and data['error_directions'].shape[1] == n_error_dirs
                        and data['coverage_directions'].shape[1]
                            == n_coverage_dirs
                        and np.allclose(data['dtheta'], theta_test,
                                        rtol=0.0, atol=1e-7)
                        and 'coverage_truth_definition' in data.files
                        and str(data['coverage_truth_definition'])
                            == 'max_k_ray_model_v1')
        if shape_ok:
            print(f'[evaluate] Use true feasible region caching: {path}')
            return path
        print(f'[evaluate] True value cache and current test conditions/The scale is inconsistent and will be recalculated.: {path}')

    rng = np.random.RandomState(SEED + 100)
    err_angles = rng.uniform(0, 2*np.pi, size=(n_test, n_error_dirs))
    cov_angles = rng.uniform(0, 2*np.pi, size=(n_test, n_coverage_dirs))
    err_dirs = np.stack((np.cos(err_angles), np.sin(err_angles)), axis=-1)
    cov_dirs = np.stack((np.cos(cov_angles), np.sin(cov_angles)), axis=-1)

    ec = case['errorcalculator'].copy()
    true_support = np.full((n_test, n_error_dirs, 2), np.nan)
    true_support_success = np.zeros((n_test, n_error_dirs), dtype=bool)
    x_ref = np.full((n_test, 2), np.nan)
    reference_success = np.zeros(n_test, dtype=bool)
    k_omega = np.full((n_test, n_coverage_dirs), np.nan)
    ray_model = create_ray_model(ec)
    reference = ReferencePointCalculator(
        ppc=ppc, init_params_dict=case['params']['params_dict'],
        original_model=ec.original_model, solver='ipopt')
    solver = ec.solver
    t0 = time.perf_counter()

    for i, dtheta in enumerate(theta_test):
        update_parameters(ec, case['params']['params_dict'], dtheta)
        for j, direction in enumerate(err_dirs[i]):
            point = ec.optimize_direction(direction)
            if point is not None and np.all(np.isfinite(point)):
                true_support[i, j] = point
                true_support_success[i, j] = True

        ref = reference.compute(dtheta)
        if ref is not None and np.all(np.isfinite(ref)):
            x_ref[i] = ref
            reference_success[i] = True
            update_ray_model_params(
                ray_model, case['params']['params_dict'], dtheta)
            for j, direction in enumerate(cov_dirs[i]):
                value = compute_k_max_ray(ray_model, ref, direction, solver)
                if value is not None and np.isfinite(value) and value > 1e-12:
                    k_omega[i, j] = value
        print(f'[truth] {i+1}/{n_test}')

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, dtheta=theta_test, error_directions=err_dirs,
             coverage_directions=cov_dirs, true_support=true_support,
             true_support_success=true_support_success, x_ref=x_ref,
             reference_success=reference_success, k_omega=k_omega,
             coverage_truth_definition=np.array('max_k_ray_model_v1'),
             computation_time=np.array(time.perf_counter()-t0))
    return path


def _point_in_polygon(point, polygon, tol=1e-10):
    """Two-dimensional ray method with boundaries counted as interior."""
    point = np.asarray(point, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = q - p
        cross = edge[0] * (point[1]-p[1]) - edge[1] * (point[0]-p[0])
        if abs(cross) <= tol and np.dot(point-p, point-q) <= tol:
            return True
    inside = False
    x, y = point
    for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (p[1] > y) != (q[1] > y):
            x_cross = p[0] + (y-p[1]) * (q[0]-p[0]) / (q[1]-p[1])
            if x < x_cross:
                inside = not inside
    return inside


def _convex_hull_polygon(points, tol=1e-12):
    """Returns the vertices of the counterclockwise convex hull of the predicted support points, removing duplicate points and collinear interior points.

    The goal of supervision is still to36original coordinate points with a fixed support direction number; the convex hull is only used to divide the network
    The output is interpreted as a geometric region to avoid small prediction errors producing concave or self-intersecting polygons..
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f'The convex hull input should be(n,2), Actually{points.shape}.')
    if not np.all(np.isfinite(points)):
        raise ValueError('The predicted support point contains non-limited values and the convex hull cannot be constructed..')
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    unique = []
    for point in ordered:
        if not unique or np.linalg.norm(point-unique[-1]) > tol:
            unique.append(point.copy())
    if len(unique) < 3:
        raise RuntimeError('Predicting support points is insufficient to form a 2D convex hull.')

    def cross(origin, first, second):
        return ((first[0]-origin[0]) * (second[1]-origin[1])
                - (first[1]-origin[1]) * (second[0]-origin[0]))

    lower = []
    for point in unique:
        while (len(lower) >= 2
               and cross(lower[-2], lower[-1], point) <= tol):
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while (len(upper) >= 2
               and cross(upper[-2], upper[-1], point) <= tol):
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        raise RuntimeError('Predict that the convex hull of the support points is degenerated and cannot form a two-dimensional polygon..')
    return hull


def _project_to_polygon(point, polygon):
    """Returns itself if the point is within the polygon, otherwise projects to the nearest line segment."""
    point = np.asarray(point, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    if _point_in_polygon(point, polygon):
        return point.copy()
    best, best_d2 = None, np.inf
    for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = q - p
        denom = float(edge @ edge)
        if denom <= 1e-20:
            candidate = p
        else:
            t = np.clip(float((point-p) @ edge) / denom, 0.0, 1.0)
            candidate = p + t * edge
        d2 = float(np.sum((point-candidate)**2))
        if d2 < best_d2:
            best, best_d2 = candidate, d2
    return best


def _max_ray_polygon_intersection(reference, direction, polygon):
    """Returns the maximum non-negative intersection distance between the ray and the ordered boundary polyline, alignedk_maxCaliber."""
    reference = np.asarray(reference, dtype=float)
    direction = np.asarray(direction, dtype=float)
    hits = []
    for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = q - p
        matrix = np.column_stack((direction, -edge))
        det = np.linalg.det(matrix)
        if abs(det) <= 1e-12:
            continue
        k, t = np.linalg.solve(matrix, p-reference)
        if k >= -1e-10 and -1e-10 <= t <= 1.0+1e-10:
            hits.append(max(float(k), 0.0))
    return max(hits) if hits else np.nan


def evaluate_model(name, model, case, truth_path, test_label_path,
                   evaluation_dir=None, online_time_override=None,
                   theta_override=None):
    """Calculate and save successive raw indicators under the same test conditions and directions."""
    truth = np.load(truth_path)
    labels = None if test_label_path is None else np.load(test_label_path)
    theta = (np.asarray(labels['dtheta'][:len(truth['dtheta'])],
                        dtype=np.float32)
             if theta_override is None else
             np.asarray(theta_override, dtype=np.float32)[:len(truth['dtheta'])])
    points_true = (np.asarray(labels['support_points'][:len(theta)],
                              dtype=np.float32)
                   if labels is not None and 'support_points' in labels.files
                   else None)
    a_true = b_true = None
    if not np.allclose(truth['dtheta'], theta, rtol=0.0, atol=1e-7):
        raise ValueError(
            f'{name}The test tags are not the same set as the true feasible region cachedtheta; '
            'Please delete the error cache or use--overwriteRecalculate.')
    points_pred_all = None
    point_polygon_all = None
    a_pred_all = b_pred_all = None
    if model is None:
        raise ValueError('Reviews are requiredmodel.')
    prediction = predict(model, theta)
    if isinstance(model, BoundaryPointNet):
        points_pred_all = prediction
        point_polygon_all = [
            _convex_hull_polygon(points) for points in points_pred_all]
    else:
        a_pred_all, b_pred_all = prediction
    # The evaluation has previously performed batch forward propagation, so timing here will yield approx.1 msafter preheating
    # Online time follows the unified definition of the first complete target-network output.
    # Generate subprocess scripts first, then write method-specific summaries.
    online_time = (np.nan if online_time_override is None
                   else float(online_time_override))
    ec = case['errorcalculator'].copy()

    shape = truth['true_support_success'].shape
    feas = np.full(shape, np.nan)
    opt = np.full(shape, np.nan)
    feas_success = np.zeros(shape, dtype=bool)
    opt_success = np.zeros(shape, dtype=bool)
    coverage = np.full(truth['k_omega'].shape, np.nan)
    reference_member = np.zeros(len(theta), dtype=bool)
    approx_direction_success = np.zeros(shape, dtype=bool)

    for i, dtheta in enumerate(theta):
        update_parameters(ec, case['params']['params_dict'], dtheta)
        point_polygon = (point_polygon_all[i]
                         if point_polygon_all is not None else None)
        if point_polygon is None:
            a_pred, b_pred = a_pred_all[i], b_pred_all[i]
            ec.update_polytope(a_pred, b_pred)

        if truth['reference_success'][i]:
            if point_polygon is not None:
                member = _point_in_polygon(truth['x_ref'][i], point_polygon)
            else:
                member, _ = reference_point_membership(
                    a_pred, b_pred, truth['x_ref'][i], REFERENCE_MEMBERSHIP_TOL)
            reference_member[i] = member
            if member:
                for j, direction in enumerate(truth['coverage_directions'][i]):
                    ko = truth['k_omega'][i, j]
                    if point_polygon is not None:
                        kp = _max_ray_polygon_intersection(
                            truth['x_ref'][i], direction, point_polygon)
                    else:
                        kp = compute_k_max_polytope(
                            a_pred, b_pred, truth['x_ref'][i], direction)
                    if np.isfinite(ko) and ko > 1e-12 and np.isfinite(kp):
                        coverage[i, j] = kp / ko

        for j, direction in enumerate(truth['error_directions'][i]):
            # Feasibility: Predict the projection distance from the polyhedral support point to the true feasible region.
            if point_polygon is not None:
                x_poly = point_polygon[np.argmin(point_polygon @ direction)]
            else:
                x_poly = ec.optimize_direction(direction, in_approx=True)
            if x_poly is not None and np.all(np.isfinite(x_poly)):
                approx_direction_success[i, j] = True
                x_true_projection = ec.project(x_poly)
                if x_true_projection is not None:
                    feas[i, j] = np.sum((x_poly-x_true_projection)**2)
                    feas_success[i, j] = True

            # Projection optimality: the projection distance from the true feasible region support point to the predicted polyhedron.
            if truth['true_support_success'][i, j]:
                if point_polygon is not None:
                    x_poly_projection = _project_to_polygon(
                        truth['true_support'][i, j], point_polygon)
                else:
                    x_poly_projection = ec.project(
                        truth['true_support'][i, j], to_approx=True)
                if x_poly_projection is not None:
                    opt[i, j] = np.sum(
                        (truth['true_support'][i, j]-x_poly_projection)**2)
                    opt_success[i, j] = True
        print(f'[evaluate {name}] {i+1}/{len(theta)}')

    # Parameter error is supplementary; geometric error and coverage remain the primary metrics.
    a_mse = (float(np.mean((a_pred_all-a_true)**2))
             if a_pred_all is not None and a_true is not None
             and np.all(np.isfinite(a_true)) else np.nan)
    b_mse = (float(np.mean((b_pred_all-b_true)**2))
             if b_pred_all is not None and b_true is not None
             and np.all(np.isfinite(b_true)) else np.nan)
    point_mse = (float(np.mean(np.sum(
        (points_pred_all-points_true)**2, axis=-1)))
                 if points_pred_all is not None and points_true is not None
                 else np.nan)
    boundary_coordinate_error = (
        points_pred_all - points_true
        if points_pred_all is not None and points_true is not None
        else np.asarray([]))
    boundary_coordinate_squared_error = boundary_coordinate_error ** 2
    hull_counts = (np.asarray([len(polygon) for polygon in point_polygon_all],
                              dtype=int)
                   if point_polygon_all is not None else np.asarray([], dtype=int))
    hull_vertices = np.full(
        (len(theta), N_SIDES, 2), np.nan, dtype=float)
    if point_polygon_all is not None:
        for index, polygon in enumerate(point_polygon_all):
            hull_vertices[index, :len(polygon)] = polygon
    valid_cov = coverage[np.isfinite(coverage)]
    valid_feas = feas[feas_success]
    valid_opt = opt[opt_success]
    metrics = {
        'method': name,
        'test_A_mse': a_mse,
        'test_b_mse': b_mse,
        'test_boundary_point_mse': point_mse,
        'convex_hull_vertex_count_mean': (
            float(np.mean(hull_counts)) if hull_counts.size else np.nan),
        'convex_hull_vertex_count_min': (
            int(np.min(hull_counts)) if hull_counts.size else np.nan),
        'feasibility_mean': float(np.mean(valid_feas)) if valid_feas.size else np.nan,
        'feasibility_p95': float(np.percentile(valid_feas, 95)) if valid_feas.size else np.nan,
        'projection_optimality_mean': float(np.mean(valid_opt)) if valid_opt.size else np.nan,
        'projection_optimality_p95': float(np.percentile(valid_opt, 95)) if valid_opt.size else np.nan,
        'coverage_mean': float(np.mean(valid_cov)) if valid_cov.size else np.nan,
        'coverage_median': float(np.median(valid_cov)) if valid_cov.size else np.nan,
        'undercoverage_mean': (float(np.mean(np.maximum(1-valid_cov, 0)))
                               if valid_cov.size else np.nan),
        'overcoverage_mean': (float(np.mean(np.maximum(valid_cov-1, 0)))
                              if valid_cov.size else np.nan),
        'reference_membership_rate': float(np.mean(reference_member)),
        'approx_direction_solve_success_rate': float(np.mean(approx_direction_success)),
        'feasibility_success_count': int(np.sum(feas_success)),
        'optimality_success_count': int(np.sum(opt_success)),
        'coverage_success_count': int(valid_cov.size),
        'online_inference_seconds': float(online_time),
    }
    output_evaluation_dir = (EVAL_DIR if evaluation_dir is None
                             else Path(evaluation_dir))
    output_evaluation_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_evaluation_dir / f'{name}_raw.npz', dtheta=theta,
             A_pred=np.asarray([]) if a_pred_all is None else a_pred_all,
             b_pred=np.asarray([]) if b_pred_all is None else b_pred_all,
             A_true=np.asarray([]) if a_true is None else a_true,
             b_true=np.asarray([]) if b_true is None else b_true,
             boundary_points_pred=(np.asarray([]) if points_pred_all is None
                                   else points_pred_all),
             boundary_points_true=(np.asarray([]) if points_true is None
                                   else points_true),
             boundary_coordinate_error=boundary_coordinate_error,
             boundary_coordinate_squared_error=boundary_coordinate_squared_error,
             boundary_polygon_vertices=hull_vertices,
             boundary_polygon_vertex_count=hull_counts,
             boundary_geometry_method=np.asarray(
                 ('convex_hull_of_36_predicted_support_points_ccw'
                  if point_polygon_all is not None else 'halfspaces_A_b')),
             A_error=(np.asarray([]) if a_pred_all is None or a_true is None
                      else a_pred_all-a_true),
             b_error=(np.asarray([]) if b_pred_all is None or b_true is None
                      else b_pred_all-b_true),
             feasibility_error=feas, projection_optimality_error=opt,
             feasibility_success=feas_success, optimality_success=opt_success,
             coverage=coverage, reference_member=reference_member,
             approx_direction_success=approx_direction_success,
             **{k: np.array(v) for k, v in metrics.items()})
    return metrics


def run_validation_physical_audit(case, ppc, label_sizes, n_validation,
                                  n_error_dirs, n_coverage_dirs,
                                  overwrite=False):
    """pair has been tagged byMSESelected supervised weights perform independent physical area audits.

    There is a strict separation between verifying ground-truth data and testing ground-truth data. The feasibility error, projected optimality error and
    Coverage is only used to check tagsMSEWhether the selected model has reasonable physical area quality is not involved.
    Gradient updates, early stopping or checkpoint selection.
    """
    audit_dir = EVAL_DIR / 'validation_physical_audit'
    validation_label_path = LABEL_DIR / 'validation_labels.npz'
    truth_path = prepare_truth(
        case, ppc, n_validation, n_error_dirs, n_coverage_dirs,
        overwrite=overwrite,
        test_label_path=validation_label_path,
        truth_path=audit_dir / 'validation_truth.npz')

    rows = []
    for n in label_sizes:
        model = load_supervised_model(case, n)
        row = evaluate_model(
            f'supervised_n{n}_validation_audit', model, case,
            truth_path, validation_label_path,
            evaluation_dir=audit_dir)
        # evaluate_modelUse the common field name of the formal test evaluation; clearly change it to
        # validation, Avoid being mistakenly thought that the test set participates in model selection.
        row['validation_A_mse'] = row.pop('test_A_mse')
        row['validation_b_mse'] = row.pop('test_b_mse')
        row['validation_boundary_point_mse'] = row.pop(
            'test_boundary_point_mse')
        rows.append(row)

    audit_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    summary_path = audit_dir / 'validation_physical_audit_summary.csv'
    with summary_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (audit_dir / 'validation_physical_audit_config.json').open(
            'w', encoding='utf-8') as f:
        json.dump({
            'case': CASENAME,
            'label_method': 'support_points',
            'label_sizes': [int(x) for x in label_sizes],
            'validation_conditions': int(n_validation),
            'error_directions_per_condition': int(n_error_dirs),
            'coverage_directions_per_condition': int(n_coverage_dirs),
            'checkpoint_selection_metric': 'validation_label_total_mse',
            'physical_audit_used_for_selection': False,
            'test_set_used': False,
            'note': (
                'Physical region metrics are computed only after the '
                'label-MSE checkpoint has been fixed.'),
        }, f, ensure_ascii=False, indent=2)
    print('[validation audit] Supervision weights are fixed; physical indicators do not participate in model selection.')
    print(f'[validation audit] Summary: {summary_path}')
    return rows


def add_cost_accounting(rows, label_sizes):
    """Supplements offline costs and training data requirements; test set generation costs are not included in any method training."""
    require_same_server_cache(LABEL_DIR / 'train_labels.npz', 'Training labels and generation timing')
    require_same_server_cache(
        LABEL_DIR / 'validation_labels.npz', 'Verify tags and generate timing')
    train_cache = np.load(LABEL_DIR / 'train_labels.npz')
    if 'cumulative_generation_time' in train_cache.files:
        train_cumulative_time = np.asarray(
            train_cache['cumulative_generation_time'], dtype=float)
    else:
        total_label_time = float(train_cache['generation_time'])
        generated_n = len(train_cache['dtheta'])
        train_cumulative_time = np.linspace(
            total_label_time/max(generated_n, 1), total_label_time, generated_n)
    validation_cache = np.load(LABEL_DIR / 'validation_labels.npz')
    validation_count = len(validation_cache['dtheta'])
    validation_time = float(validation_cache['generation_time'])

    if ('cumulative_oracle_calls' not in train_cache.files
            or 'cumulative_oracle_calls' not in validation_cache.files):
        raise ValueError('Support point label cache missing cumulative_oracle_calls.')
    train_cumulative_calls = np.asarray(
        train_cache['cumulative_oracle_calls'], dtype=int)
    validation_calls = int(np.asarray(
        validation_cache['cumulative_oracle_calls'], dtype=int)[-1])

    if not PINN_TRAINING_METRICS.exists():
        raise FileNotFoundError(
            f'Can’t find same serverPINNTraining timing: {PINN_TRAINING_METRICS}\n'
            'Please run first --stage pinn.')
    pinn_meta = np.load(PINN_TRAINING_METRICS)
    pinn_train_time = float(pinn_meta['total_train_time'])
    pinn_pre_steps = int(pinn_meta['pretrain_optimizer_steps'])
    pinn_full_steps = int(pinn_meta['fullnet_optimizer_steps'])
    pinn_full_condition_exposures = int(pinn_meta['fullnet_training_samples'])
    pinn_direction_probes = (
        int(pinn_meta['pretrain_direction_probes'])
        + int(pinn_meta['fullnet_direction_probes']))
    pinn_oracle_calls = (
        int(pinn_meta['pretrain_true_domain_oracle_calls'])
        + int(pinn_meta['fullnet_true_domain_oracle_calls']))

    for row in rows:
        method = row['method']
        if method == 'pinn':
            row.update({
                'full_region_training_labels': 0,
                'full_region_validation_labels': 0,
                'total_offline_full_region_labels': 0,
                # CaseData.__getitem__ Re-randomly generated for each visit dtheta; under continuous distribution
    # Treat the 12,100 oracle calls as distinct physical training observations.
                'unique_training_conditions': pinn_full_condition_exposures,
                'labelled_boundary_directions': 0,
                'online_physics_direction_probes': pinn_direction_probes,
                'physics_oracle_calls': pinn_oracle_calls,
                'optimizer_steps': pinn_pre_steps + pinn_full_steps,
                'min_optimizer_steps': np.nan,
                'max_optimizer_steps': np.nan,
                'best_validation_step': np.nan,
                'best_validation_loss': np.nan,
                'stopping_reason': 'fixed_two_stage_schedule',
                'converged_by_validation': False,
                'label_generation_seconds': 0.0,
                'network_training_seconds': pinn_train_time,
                'offline_total_seconds': pinn_train_time,
                'condition_exposures': pinn_full_condition_exposures,
            })
        else:
            n = int(method.split('_n')[-1])
            train_meta = np.load(MODEL_DIR / f'n{n}' / 'training_metrics.npz')
            convergence_fields = {
                'min_optimizer_steps', 'max_optimizer_steps',
                'best_validation_step', 'stopping_reason',
                'converged_by_validation',
            }
            missing = convergence_fields.difference(train_meta.files)
            if missing:
                raise RuntimeError(
                    f'supervised_n{n}It is still the result of the old supervised training, lacking'
                    f'{sorted(missing)}.Please run first--stage trainor--stage all, '
                    'Retrain with new validation convergence protocol.')
            label_time = float(train_cumulative_time[n-1]) + validation_time
            train_time = float(train_meta['train_time'])
            physics_calls = int(train_cumulative_calls[n-1]) + validation_calls
            row.update({
                'full_region_training_labels': n,
                'full_region_validation_labels': validation_count,
                'total_offline_full_region_labels': n + validation_count,
                'unique_training_conditions': n,
                'labelled_boundary_directions': n * N_SIDES,
                'online_physics_direction_probes': 0,
                # Offline calculations include validation tags for model selection; independent test tags are not included.
                'physics_oracle_calls': physics_calls,
                'optimizer_steps': int(train_meta['optimizer_steps']),
                'min_optimizer_steps': int(train_meta['min_optimizer_steps']),
                'max_optimizer_steps': int(train_meta['max_optimizer_steps']),
                'best_validation_step': int(train_meta['best_validation_step']),
                'best_validation_loss': float(
                    train_meta['best_validation_loss']),
                'stopping_reason': str(train_meta['stopping_reason']),
                'converged_by_validation': bool(
                    train_meta['converged_by_validation']),
                'label_generation_seconds': label_time,
                'network_training_seconds': train_time,
                'offline_total_seconds': label_time + train_time,
                'condition_exposures': int(train_meta['optimizer_steps']) * BATCH_SIZE,
            })
    return rows


SUMMARY_FIELDS = [
    'method', 'full_region_training_labels', 'full_region_validation_labels',
    'total_offline_full_region_labels', 'unique_training_conditions',
    'labelled_boundary_directions', 'online_physics_direction_probes',
    'condition_exposures',
    'physics_oracle_calls', 'optimizer_steps',
    'min_optimizer_steps', 'max_optimizer_steps',
    'best_validation_step', 'best_validation_loss',
    'stopping_reason', 'converged_by_validation',
    'label_generation_seconds',
    'network_training_seconds', 'offline_total_seconds',
    'online_inference_seconds', 'test_A_mse', 'test_b_mse',
    'test_boundary_point_mse', 'convex_hull_vertex_count_mean',
    'convex_hull_vertex_count_min',
    'feasibility_mean', 'feasibility_p95',
    'projection_optimality_mean', 'projection_optimality_p95',
    'coverage_mean', 'coverage_median', 'undercoverage_mean',
    'overcoverage_mean', 'reference_membership_rate',
    'approx_direction_solve_success_rate', 'feasibility_success_count',
    'optimality_success_count', 'coverage_success_count',
]


def apply_unified_online_times(rows, required=False):
    """Write unified online timing according to the method name to avoid mixing in the post-warm-up time of the evaluation process."""
    timing = load_unified_online_times(required=required)
    for row in rows:
        row['online_inference_seconds'] = timing.get(row['method'], np.nan)
    return rows


def synchronize_existing_online_times(required=True):
    """There is no need to re-run the physical evaluation, the correction has already been madesummaryandrawOld timing fields in file."""
    summary_path = SUPPORT_OUT / 'summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f'not foundR1.6Formal summary: {summary_path}')
    with summary_path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    apply_unified_online_times(rows, required=required)
    with summary_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    timing = load_unified_online_times(required=required)
    for method, elapsed in timing.items():
        raw_path = SUPPORT_OUT / 'evaluation' / f'{method}_raw.npz'
        if not raw_path.exists():
            continue
        with np.load(raw_path, allow_pickle=False) as old:
            payload = {key: old[key] for key in old.files}
        payload['online_inference_seconds'] = np.asarray(elapsed)
        payload['online_timing_source'] = np.asarray(
            repository_relative_path(UNIFIED_ONLINE_TIMING_CSV))
        np.savez(raw_path, **payload)
    print(f'[online timing] Synchronized and unified original caliber: {summary_path}')


def save_summary(rows, label_sizes=None):
    if label_sizes is None:
        label_sizes = LABEL_SIZES
    apply_unified_online_times(rows, required=False)
    EXPERIMENT_OUT.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT_OUT / 'summary.csv').open(
        'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    label_definition = (
        '36 ordered support points argmax d_i^T(P,Q) on the original AC '
        'feasible domain; network directly predicts their coordinates')
    with (EXPERIMENT_OUT / 'experiment_config.json').open(
            'w', encoding='utf-8') as f:
        json.dump({
            'case': CASENAME, 'n_sides': N_SIDES,
            'label_sizes': [int(x) for x in label_sizes],
            'batch_size': BATCH_SIZE,
            'min_optimizer_steps': MIN_OPTIMIZER_STEPS,
            'max_optimizer_steps': MAX_OPTIMIZER_STEPS,
            'validation_interval': VALIDATION_INTERVAL,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'relative_improvement_tolerance': RELATIVE_IMPROVEMENT_TOL,
            'absolute_improvement_tolerance': ABSOLUTE_IMPROVEMENT_TOL,
            'checkpoint_selection_metric': 'validation_label_total_mse',
            'physical_audit_used_for_selection': False,
            'learning_rate': LEARNING_RATE,
            'device': str(DEVICE),
            'machine_signature': json.loads(MACHINE_SIGNATURE),
            'label_method': 'support_points',
            'label_definition': label_definition,
            'support_label_cache_method': SUPPORT_LABEL_METHOD,
            'support_cache_schema_version': SUPPORT_CACHE_SCHEMA_VERSION,
            'support_cache_signature': SUPPORT_CACHE_SIGNATURE,
            'support_point_search': {
                'direction_normalization': 'unit Euclidean norm',
                'objective': 'argmax direction^T [P,Q]',
                'implementation': (
                    'ErrorCalculator.optimize_direction(-direction, '
                    'in_approx=False)'),
                'physical_domain': 'original AC feasible domain',
            },
            'predicted_region_construction': (
                'compute the convex hull of the 36 predicted support points; '
                'connect retained hull vertices in counter-clockwise hull order'),
            'label_mse_uses_raw_ordered_points': True,
            'physical_evaluation_uses_convex_hull': True,
            'optimality_definition': 'geometric projection error, not objective regret',
            'online_timing_source': repository_relative_path(
                UNIFIED_ONLINE_TIMING_CSV),
            'online_timing_protocol': (
                'manuscript order; independent process; model loading and '
                'PreTrainNet initialization excluded; target network not warmed; '
                'one first complete region output'),
        }, f, ensure_ascii=False, indent=2)
    print(f'Summary saved: {EXPERIMENT_OUT / "summary.csv"}')


def run_labels(case, ppc, train_count, val_count, test_count, overwrite=False):
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    common_test_conditions = load_common_test_conditions(
        case['params']['count'], overwrite=False)[:test_count]
    generate_support_label_split(
        'train', train_count, SEED, case, ppc, overwrite)
    generate_support_label_split(
        'validation', val_count, SEED+1, case, ppc, overwrite)
    generate_support_label_split(
        'test', test_count, SEED+2, case, ppc, overwrite,
        fixed_conditions=common_test_conditions)


def main():
    parser = argparse.ArgumentParser(description='R1.6 Support point supervised learning baseline')
    parser.add_argument('--stage',
                        choices=['all', 'labels', 'pinn', 'train', 'evaluate',
                                 'sync_online_time'],
                        default=RUN_STAGE,
                        help=f'Running phase; current default{RUN_STAGE}')
    parser.add_argument('--quick', action='store_true', help='Small-scale process inspection')
    parser.add_argument('--overwrite', action='store_true', help='Recalculate existing cache/weight')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    if args.quick:
        label_sizes = (10, 20)
        n_val, n_test, max_steps = 5, 5, 20
        n_err, n_cov = 3, 5
    else:
        label_sizes = LABEL_SIZES
        n_val, n_test, max_steps = N_VALIDATION, N_TEST, MAX_OPTIMIZER_STEPS
        n_err, n_cov = N_ERROR_DIRECTIONS, N_COVERAGE_DIRECTIONS

    print(f'R1.6 supervised baseline | case={CASENAME} | device={DEVICE}')
    print(f'label method=support_points, Results directory={EXPERIMENT_OUT}')
    if args.quick:
        training_budget_text = f'most{max_steps}step (quick) '
    else:
        training_budget_text = f'The verification platform stops early, at most{max_steps}step'
    print(f'Label volume={label_sizes}, Verify={n_val}, test={n_test}, '
          f'Supervised training={training_budget_text}')
    if args.stage == 'sync_online_time':
        synchronize_existing_online_times(required=True)
        return
    ppc, case = build_case()

    if args.stage in ('all', 'labels'):
        run_labels(case, ppc, max(label_sizes), n_val, n_test, args.overwrite)
    if args.stage in ('all', 'pinn'):
        train_pinn_same_server(ppc, case, args.overwrite, quick=args.quick)
    if args.stage in ('all', 'train'):
        for n in label_sizes:
            train_one(n, max_steps, args.lr, case, args.overwrite)
        run_validation_physical_audit(
            case, ppc, label_sizes, n_val, n_err, n_cov,
            overwrite=args.overwrite)
    if args.stage in ('all', 'evaluate'):
        truth_path = prepare_truth(case, ppc, n_test, n_err, n_cov,
                                   args.overwrite)
        rows = []
        pinn = load_pinn_model(case)
        rows.append(evaluate_model(
            'pinn', pinn, case, truth_path, LABEL_DIR/'test_labels.npz'))
        for n in label_sizes:
            model = load_supervised_model(case, n)
            rows.append(evaluate_model(
                f'supervised_n{n}', model, case, truth_path,
                LABEL_DIR/'test_labels.npz'))
        rows = add_cost_accounting(rows, label_sizes)
        save_summary(rows)


if __name__ == '__main__':
    main()
