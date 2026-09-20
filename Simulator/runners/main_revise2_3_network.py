# -*- coding: utf-8 -*-
"""Run the R2.3 hidden-layer and activation-function comparison."""
import os
import time
import csv
import argparse
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # avoid PyTorch OpenMP Repeated loading (same as main_ds.py:5)

from Simulator.Approximator import PreTrainNet, FullNet, compute_loss, Trainer
from Simulator.cases import TD_case
from Simulator import PROJECT_ROOT
from Simulator.reference_point import (
    REFERENCE_MEMBERSHIP_TOL,
    reference_point_membership,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def k_max_polytope(A, b, x_ref, direction):
    """Polyhedron Ax<=b from the reference point along direction The maximum radial length of."""
    av = A @ direction
    slack = b - A @ x_ref
    valid = av > 1e-12
    if not np.any(valid):
        return np.nan
    k = np.min(slack[valid] / av[valid])
    return float(k) if np.isfinite(k) else np.nan


def compute_errors_random_dtheta_safe(error_calculator, nn_model, ppc, device,
                                      init_params_dict, n_dtheta=5,
                                      n_samples_per_dtheta=10, zero_dtheta=False):
    """randomly generated Δθ, Compute feasibility and optimality errors.
    Return from main_ds.py:15 (main_ds.py Executing scripts for the top level is not possible import, Therefore copied here).
    use error_calculator.copy() Does not contaminate the original object; np.random.seed(42) Fixed evaluation sample,
    Put different configurations in the same group Δθ On the evaluation, the errors are directly comparable.
    """
    error_calc_copy = error_calculator.copy()
    # Coverage follows the radial definition of the project’s unified reference point rho=k_P/k_Omega.
    error_calc_copy.attach_radial(ppc, init_params_dict,
                                  n_dirs=n_samples_per_dtheta, seed=123)
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    total_params = Pd_init.size + Qd_init.size
    dtheta_range = (-0.5, 0.5)
    np.random.seed(42)
    if zero_dtheta:
        dtheta_values = [np.zeros(total_params)]
    else:
        dtheta_values = [np.random.uniform(dtheta_range[0], dtheta_range[1], total_params)
                         for _ in range(n_dtheta)]
    feas_errors, opt_errors, coverage_values = [], [], []
    coverage_rng = np.random.RandomState(123)
    coverage_angles = coverage_rng.uniform(
        0.0, 2.0 * np.pi, size=(len(dtheta_values), n_samples_per_dtheta))
    coverage_dirs = np.stack(
        (np.cos(coverage_angles), np.sin(coverage_angles)), axis=-1)

    for dtheta_idx, dtheta in enumerate(dtheta_values):
        Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
        Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)
        error_calc_copy.update_parameters({'Pd_meta': Pd_meta_new, 'Qd_meta': Qd_meta_new})
        with torch.no_grad():
            dtheta_tensor = torch.tensor(dtheta, dtype=torch.float32).to(device)
            if isinstance(nn_model, FullNet):
                A_pred, b_pred = nn_model(dtheta_tensor)
            elif isinstance(nn_model, PreTrainNet):
                A_pred, b_pred = nn_model()
                A_pred = A_pred.repeat(1, 1, 1)
                b_pred = b_pred.repeat(1, 1)
            else:
                raise ValueError(f"Unsupported model type: {type(nn_model)}")
            A_pred_np = A_pred[0].detach().cpu().numpy()
            b_pred_np = b_pred[0].detach().cpu().numpy()
        error_calc_copy.update_polytope(A_hat=A_pred_np, b_hat=b_pred_np)
        feas_results, opt_results = error_calc_copy.calculate(
            n_cal=n_samples_per_dtheta, cal_feas=True, cal_opt=True)
        feas_errors.extend([r['error'] for r in feas_results])
        opt_errors.extend([r['error'] for r in opt_results])

        # Mark the condition as failed when its reference point lies outside the predicted polyhedron.
        radial = error_calc_copy.calculate_radial(
            coverage_dirs[dtheta_idx], dtheta)
        if radial is None:
            coverage_values.extend([np.nan] * n_samples_per_dtheta)
            continue
        x_ref = radial[0]['x_ref']
        is_member, _ = reference_point_membership(
            A_pred_np, b_pred_np, x_ref, REFERENCE_MEMBERSHIP_TOL)
        if not is_member:
            coverage_values.extend([np.nan] * n_samples_per_dtheta)
            continue
        for rec, direction in zip(radial, coverage_dirs[dtheta_idx]):
            k_omega = float(rec['k_omega'])
            k_poly = k_max_polytope(
                A_pred_np, b_pred_np, x_ref, direction)
            if k_omega > 1e-9 and np.isfinite(k_poly):
                coverage_values.append(k_poly / k_omega)
            else:
                coverage_values.append(np.nan)
    return (np.array(feas_errors), np.array(opt_errors),
            np.array(coverage_values))


# ===== Experiment configuration matrix (6 Configuration) =====
# fixed width 128, Structures only compare single hidden layer/Dual hidden layers, activations compare only relu/tanh/silu.
CONFIGS = [
    {'hidden_sizes': [128],      'activation': 'relu'},
    {'hidden_sizes': [128],      'activation': 'tanh'},
    {'hidden_sizes': [128],      'activation': 'silu'},
    {'hidden_sizes': [128, 128], 'activation': 'relu'},
    {'hidden_sizes': [128, 128], 'activation': 'tanh'},
    {'hidden_sizes': [128, 128], 'activation': 'silu'},
]

CASENAME = 'case33bw_ds'
N_TRAIN_FULL = 20     # replica main_ds.py fullnet of n_train (phase1=4*n-1, phase2=2*n)
N_TRAIN_PRE = 500     # replica main_ds.py pretrainnet of n_train (4*n)
N_EVAL_DTHETA = 20
N_EVAL_DIRECTIONS = 36
EVAL_SEED = 42

# ===== run control (like main_ds.py top model_type Switch here as well; command line --stage/--only Coverable) =====
STAGE = 'fullnet'    # 'pretrain'=training PreTrainNet concurrent disk; 'fullnet'=load weight training FullNet Configuration
ONLY = None          # fullnet only run config tag (comma separated) ; None=All 6 a

# ===== path =====
OUT = PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison' / 'network' / CASENAME
PRETRAIN_DIR = OUT / 'pretrain'
PRETRAIN_WEIGHTS = PRETRAIN_DIR / 'pretrainnet_weights.pth'   # PreTrainNet weight storage location

SUMMARY_FIELDS = ['config', 'hidden_sizes', 'activation',
                  'feas_mean', 'feas_median', 'opt_mean', 'opt_median',
                  'coverage_mean', 'coverage_median', 'coverage_count',
                  'coverage_failed', 'train_time']
FORMAL_EVAL_DIR = OUT / 'formal_evaluation'
FORMAL_SUMMARY_FIELDS = [
    'config', 'hidden_sizes', 'activation', 'parameter_count',
    'n_dtheta', 'n_directions',
    'feas_mean', 'feas_median', 'feas_p95',
    'opt_mean', 'opt_median', 'opt_p95',
    'coverage_mean', 'coverage_median', 'coverage_p05', 'coverage_p95',
    'undercoverage_mean', 'undercoverage_p95', 'overcoverage_mean',
    'membership_failed_cases', 'true_support_success_rate',
    'radial_success_rate', 'feas_success_rate', 'opt_success_rate',
    'phase2_train_time_s']


def activation_str(cfg):
    """Convert activation specifications to string labels; all six official configurations use unified activation."""
    a = cfg['activation']
    return '-'.join(a) if isinstance(a, list) else a


def config_tag(cfg):
    h = '-'.join(str(x) for x in cfg['hidden_sizes'])
    return f"h{h}_act-{activation_str(cfg)}"


# ============================================================
#  stage 1: PreTrainNet —— training warm-start Starting point and saving disk
# ============================================================
def run_pretrain():
    ppc = TD_case.case33bw_ds()
    P_rated = sum(ppc['bus'][:, 2]) / ppc['baseMVA']
    print(f"[pretrain] Equipment: {device}  |  P_rated={P_rated:.4f}  |  n_train={N_TRAIN_PRE*4}")

    # Reuse the main_ds.py training-process plot and write it to the revision output directory.
    case_pre = TD_case.DScase_train(casedata=ppc, model_type='pretrainnet',
                                    plot_flag=True, device=device,
                                    figure_dir=str(PRETRAIN_DIR / 'figures'))
    pre_model = PreTrainNet(case_pre['A_hat'], case_pre['b_hat'],
                            is_epigraph=False, device=device)
    pre_trainer = Trainer(model=pre_model,
                          error_calculator=case_pre['errorcalculator'],
                          compute_loss=compute_loss)
    pre_trainer.configure(**case_pre['trainer_configure'])   # Contains DScase_train drawing callback (draw figure_dir designated revise1)
    pre_trainer.configure(lr=1e-1 / P_rated)          # replica main_ds.py:113,207
    pre_trainer.configure(rate_opt_feas=0.6)          # replica main_ds.py:115,208
    pre_trainer.initialize()

    t0 = time.time()
    pre_trainer.train(n_train=N_TRAIN_PRE * 4, params_data=case_pre['params'])   # 2000
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(pre_model.state_dict(), PRETRAIN_WEIGHTS)
    print(f"[pretrain] completed, time consuming {time.time()-t0:.1f}s")
    print(f"[pretrain] The weight has been saved: {PRETRAIN_WEIGHTS}")
    print("[pretrain] Can be run next: python -m Simulator.runners.main_revise2_3_network --stage fullnet ...")


# ============================================================
#  stage 2: FullNet —— load Saved PreTrainNet, Training configurations + fair assessment
# ============================================================
def run_fullnet(configs):
    if not PRETRAIN_WEIGHTS.exists():
        raise FileNotFoundError(
            f"not found PreTrainNet weight {PRETRAIN_WEIGHTS}, Please run first: "
            f"python -m Simulator.runners.main_revise2_3_network --stage pretrain")

    ppc = TD_case.case33bw_ds()
    P_rated = sum(ppc['bus'][:, 2]) / ppc['baseMVA']
    case_full = TD_case.DScase_train(casedata=ppc, model_type='fullnet',
                                     plot_flag=False, device=device)
    dim_theta = case_full['params']['count']
    init_params_dict = case_full['params']['params_dict']
    error_calculator = case_full['errorcalculator']

    # ---- like main_ds.py:181-186 Same, rebuild PreTrainNet and load Save weight ----
    pre_model = PreTrainNet(case_full['A_hat'], case_full['b_hat'],
                            is_epigraph=False, device=device)
    pre_model.load_state_dict(torch.load(PRETRAIN_WEIGHTS, map_location=device))
    A_pre, b_pre = pre_model()
    A_pre_np = A_pre[0].detach().cpu().numpy()
    b_pre_np = b_pre[0].detach().cpu().numpy()
    print(f"[fullnet] Loaded PreTrainNet weight: {PRETRAIN_WEIGHTS}")
    print(f"[fullnet] Ready to run configuration {len(configs)} a: {[config_tag(c) for c in configs]}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for idx, cfg in enumerate(configs, 1):
        tag = config_tag(cfg)
        print(f"  ({idx}/{len(configs)}) {tag} ...", flush=True)
        torch.manual_seed(0)      # Fixed hidden layer random initialization vs. dataloader shuffle, Ensure that configurations are comparable
        np.random.seed(0)
        model = FullNet(dim_theta, A_init=A_pre_np, b_init=b_pre_np,
                        hidden_sizes=cfg['hidden_sizes'],
                        activation=cfg['activation'], device=device).to(device)
        trainer = Trainer(model=model, error_calculator=error_calculator,
                          compute_loss=compute_loss)
        trainer.configure(**case_full['trainer_configure'])   # Contains DScase_train of callback: Print every round Iter X: FeasErr/OptErr
        # Phase1: Comprehensive (rate_opt_feas=0.6) —— replica main_ds.py:237-240
        trainer.configure(lr=3e-5 / P_rated, rate_opt_feas=0.6)
        trainer.initialize()
        trainer.train(n_train=N_TRAIN_FULL * 4 - 1, params_data=case_full['params'])   # 79
        # Phase2: Focus on feasibility (lr very small, rate_opt_feas Very small)——Replica main_ds.py:246-251
        trainer.configure(lr=1e-5 / P_rated, rate_opt_feas=1e-4)
        trainer.initialize()
        t_p2 = time.time()
        trainer.train(n_train=N_TRAIN_FULL * 2, params_data=case_full['params'])       # 40
        phase2_time = time.time() - t_p2

        # Assessment: Fixed Δθ (internal seed=42 + copy()) , Direct comparisons across configurations
        feas_err, opt_err, coverage = compute_errors_random_dtheta_safe(
            error_calculator, model, ppc, device, init_params_dict,
            n_dtheta=5, n_samples_per_dtheta=10)

        coverage_valid = coverage[np.isfinite(coverage)]
        coverage_mean = (float(coverage_valid.mean())
                         if coverage_valid.size else np.nan)
        coverage_median = (float(np.median(coverage_valid))
                           if coverage_valid.size else np.nan)

        # Save this configuration (weights + error distribution + time)
        cfg_dir = OUT / tag
        cfg_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), cfg_dir / 'fullnet_weights.pth')
        np.savez(cfg_dir / 'eval.npz', feas=feas_err, opt=opt_err,
                 coverage=coverage, train_time=phase2_time,
                 coverage_definition=np.array('reference_point_radial_kP_over_kOmega'))

        summary_rows.append({
            'config': tag,
            'hidden_sizes': ','.join(str(x) for x in cfg['hidden_sizes']),
            'activation': activation_str(cfg),
            'feas_mean': float(feas_err.mean()),
            'feas_median': float(np.median(feas_err)),
            'opt_mean': float(opt_err.mean()),
            'opt_median': float(np.median(opt_err)),
            'coverage_mean': coverage_mean,
            'coverage_median': coverage_median,
            'coverage_count': int(coverage_valid.size),
            'coverage_failed': int(coverage.size - coverage_valid.size),
            'train_time': float(phase2_time),
        })
        print(f"        feas_mean={feas_err.mean():.3e}  opt_mean={opt_err.mean():.3e}  "
              f"coverage_mean={coverage_mean:.4f} "
              f"coverage_valid={coverage_valid.size}/{coverage.size}  "
              f"phase2_time={phase2_time:.1f}s")

    # ---- Summary: with existing summary Merge (same as config Coverage update, different config Accumulate) ----
    merged = load_existing_summary()
    for r in summary_rows:
        merged[r['config']] = r
    save_summary(merged)
    print(f"[fullnet] Summary updated (total {len(merged)} configuration) : {OUT / 'summary.csv'}")
    print("[fullnet] Done. run R2_3_network_comparison_plot.py Comparison chart can be generated.")


def load_existing_summary():
    """Read the existing results of six official configurations; automatically exclude old redundant configurations."""
    path = OUT / 'summary.csv'
    if not path.exists():
        return {}
    allowed = {config_tag(c) for c in CONFIGS}
    with open(path, 'r', encoding='utf-8') as f:
        return {r['config']: r for r in csv.DictReader(f)
                if r['config'] in allowed}


def save_summary(rows_dict):
    """write out summary.csv + summary.npz (Unified conversion of numerical fields float) ."""
    rows = list(rows_dict.values())
    float_fields = ['feas_mean', 'feas_median', 'opt_mean', 'opt_median',
                    'coverage_mean', 'coverage_median', 'train_time']
    int_fields = ['coverage_count', 'coverage_failed']
    with open(OUT / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            row = dict(r)
            # Compatible with the old ones that existed before the coverage indicator was added summary OK.
            for k in float_fields:
                row[k] = float(row.get(k, np.nan))
            for k in int_fields:
                value = row.get(k, 0)
                row[k] = int(float(value)) if str(value).strip() else 0
            w.writerow(row)
    np.savez(OUT / 'summary.npz',
             config=np.array([r['config'] for r in rows]),
             hidden_sizes=np.array([r['hidden_sizes'] for r in rows]),
             activation=np.array([r['activation'] for r in rows]),
             feas_mean=np.array([float(r['feas_mean']) for r in rows]),
             feas_median=np.array([float(r['feas_median']) for r in rows]),
             opt_mean=np.array([float(r['opt_mean']) for r in rows]),
             opt_median=np.array([float(r['opt_median']) for r in rows]),
             coverage_mean=np.array([
                 float(r.get('coverage_mean', np.nan)) for r in rows]),
             coverage_median=np.array([
                 float(r.get('coverage_median', np.nan)) for r in rows]),
             coverage_count=np.array([
                 int(float(r.get('coverage_count', 0) or 0)) for r in rows]),
             coverage_failed=np.array([
                 int(float(r.get('coverage_failed', 0) or 0)) for r in rows]),
             train_time=np.array([float(r['train_time']) for r in rows]))


def _formal_directions(n_directions=N_EVAL_DIRECTIONS):
    angle = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)
    return np.column_stack((np.cos(angle), np.sin(angle)))


def _formal_dthetas(dim_theta, n_dtheta=N_EVAL_DTHETA):
    rng = np.random.RandomState(EVAL_SEED)
    values = rng.uniform(-0.5, 0.5, size=(n_dtheta, dim_theta))
    values[0] = 0.0
    return values


def _update_eval_parameters(ec, init_params_dict, dtheta):
    pd_init = init_params_dict['Pd_meta']['initial_value']
    qd_init = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd_init.size
    ec.update_parameters({
        'Pd_meta': pd_init + dtheta[:n_pd].reshape(pd_init.shape),
        'Qd_meta': qd_init + dtheta[n_pd:].reshape(qd_init.shape),
    })


def _finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict(mean=np.nan, median=np.nan, p05=np.nan, p95=np.nan)
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p05': float(np.percentile(values, 5)),
        'p95': float(np.percentile(values, 95)),
    }


def _polygon_vertices(A, b, tol=1e-7):
    """Enumerate two-dimensional half-space intersection points and return the convex polygon vertices arranged by polar angles."""
    vertices = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            matrix = np.vstack((A[i], A[j]))
            if abs(np.linalg.det(matrix)) <= 1e-12:
                continue
            point = np.linalg.solve(matrix, np.asarray([b[i], b[j]]))
            if np.all(A @ point <= b + tol):
                vertices.append(point)
    if len(vertices) < 3:
        return np.empty((0, 2))
    vertices = np.unique(np.round(np.asarray(vertices), 12), axis=0)
    if len(vertices) < 3:
        return np.empty((0, 2))
    center = np.mean(vertices, axis=0)
    angle = np.arctan2(vertices[:, 1] - center[1],
                       vertices[:, 0] - center[0])
    return vertices[np.argsort(angle)]


def _polytope_direction_point(vertices, direction):
    if len(vertices) < 3:
        return None
    return vertices[int(np.argmin(vertices @ direction))].copy()


def _project_to_polygon(A, b, vertices, target, tol=1e-8):
    """Exact Euclidean projection of a 2D convex polygon; interior points are returned unchanged."""
    target = np.asarray(target, dtype=float)
    if len(vertices) < 3 or not np.all(np.isfinite(target)):
        return None
    if np.all(A @ target <= b + tol):
        return target.copy()
    best, best_distance = None, np.inf
    for i in range(len(vertices)):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % len(vertices)]
        edge = p1 - p0
        denom = float(edge @ edge)
        if denom <= 1e-18:
            candidate = p0
        else:
            fraction = float(np.clip((target - p0) @ edge / denom, 0.0, 1.0))
            candidate = p0 + fraction * edge
        distance = float(np.sum((target - candidate) ** 2))
        if distance < best_distance:
            best_distance = distance
            best = candidate.copy()
    return best


def _prepare_case33_truth(error_calculator, ppc, init_params_dict,
                          dthetas, eval_dirs):
    """Calculate oncecase33Common true feasible region data for strict pairing and reuse of six structures."""
    FORMAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cache = FORMAL_EVAL_DIR / 'case33_truth_cache.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=True) as data:
            if (np.array_equal(data['dthetas'], dthetas)
                    and np.array_equal(data['eval_dirs'], eval_dirs)):
                print(f'[evaluate/truth] Usecase33common true feasible region cache: {cache}')
                return {key: data[key].copy() for key in data.files}
        print('[evaluate/truth] Cache operating conditions or directions are inconsistent, recalculate.')

    ec = error_calculator.copy()
    ec.attach_radial(ppc, init_params_dict, n_dirs=len(eval_dirs), seed=123)
    shape = (len(dthetas), len(eval_dirs))
    true_support = np.full(shape + (2,), np.nan)
    true_support_success = np.zeros(shape, dtype=bool)
    x_ref = np.full((len(dthetas), 2), np.nan)
    k_omega = np.full(shape, np.nan)
    radial_success = np.zeros(shape, dtype=bool)

    for i, dtheta in enumerate(dthetas):
        _update_eval_parameters(ec, init_params_dict, dtheta)
        for j, direction in enumerate(eval_dirs):
            point = ec.optimize_direction(direction, in_approx=False)
            if point is not None and np.all(np.isfinite(point)):
                true_support[i, j] = point
                true_support_success[i, j] = True

        radial = ec.calculate_radial(eval_dirs, dtheta)
        if radial is not None:
            x_ref[i] = radial[0]['x_ref']
            for j, record in enumerate(radial):
                value = float(record['k_omega'])
                if np.isfinite(value) and value > 1e-9:
                    k_omega[i, j] = value
                    radial_success[i, j] = True
        print(
            f'[evaluate/truth] {i + 1}/{len(dthetas)}: '
            f'support={true_support_success[i].sum()}/{len(eval_dirs)}, '
            f'radial={radial_success[i].sum()}/{len(eval_dirs)}')

    np.savez(
        cache, case=np.asarray(CASENAME), dthetas=dthetas,
        eval_dirs=eval_dirs, true_support=true_support,
        true_support_success=true_support_success, x_ref=x_ref,
        k_omega=k_omega, radial_success=radial_success,
        seed=np.asarray(EVAL_SEED))
    print(f'[evaluate/truth] saved: {cache}')
    return {
        'case': np.asarray(CASENAME), 'dthetas': dthetas,
        'eval_dirs': eval_dirs, 'true_support': true_support,
        'true_support_success': true_support_success, 'x_ref': x_ref,
        'k_omega': k_omega, 'radial_success': radial_success,
        'seed': np.asarray(EVAL_SEED),
    }


def _evaluate_existing_config(cfg, case_full, ppc, truth, old_time):
    tag = config_tag(cfg)
    weight_path = OUT / tag / 'fullnet_weights.pth'
    if not weight_path.exists():
        raise FileNotFoundError(f'Trained not foundcase33weight: {weight_path}')

    model = FullNet(
        case_full['params']['count'], A_init=case_full['A_hat'],
        b_init=case_full['b_hat'], hidden_sizes=cfg['hidden_sizes'],
        activation=cfg['activation'], device=device).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    parameter_count = int(sum(p.numel() for p in model.parameters()))

    dthetas = truth['dthetas']
    eval_dirs = truth['eval_dirs']
    shape = (len(dthetas), len(eval_dirs))
    feasibility = np.full(shape, np.nan)
    optimality = np.full(shape, np.nan)
    coverage = np.full(shape, np.nan)
    k_poly = np.full(shape, np.nan)
    feasibility_success = np.zeros(shape, dtype=bool)
    optimality_success = np.zeros(shape, dtype=bool)
    membership_failure = np.zeros(len(dthetas), dtype=bool)
    polygon_valid = np.zeros(len(dthetas), dtype=bool)
    predicted_A = np.full((len(dthetas), 36, 2), np.nan)
    predicted_b = np.full((len(dthetas), 36), np.nan)

    ec = case_full['errorcalculator'].copy()
    init_params = case_full['params']['params_dict']
    start = time.time()
    for i, dtheta in enumerate(dthetas):
        _update_eval_parameters(ec, init_params, dtheta)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                dtheta, dtype=torch.float32, device=device)
            A_tensor, b_tensor = model(tensor)
        A = A_tensor[0].detach().cpu().numpy()
        b = b_tensor[0].detach().cpu().numpy()
        predicted_A[i] = A
        predicted_b[i] = b
        ec.update_polytope(A_hat=A, b_hat=b)
        vertices = _polygon_vertices(A, b)
        polygon_valid[i] = len(vertices) >= 3

        reference = truth['x_ref'][i]
        is_member, _ = reference_point_membership(
            A, b, reference, REFERENCE_MEMBERSHIP_TOL)
        membership_failure[i] = not is_member

        for j, direction in enumerate(eval_dirs):
            x_poly = _polytope_direction_point(vertices, direction)
            x_projected_true = ec.project(x_poly) if x_poly is not None else None
            if (x_poly is not None and x_projected_true is not None
                    and np.all(np.isfinite(x_poly))
                    and np.all(np.isfinite(x_projected_true))):
                feasibility[i, j] = np.sum(
                    (np.asarray(x_poly) - np.asarray(x_projected_true)) ** 2)
                feasibility_success[i, j] = True

            if truth['true_support_success'][i, j]:
                x_true = truth['true_support'][i, j]
                x_projected_poly = _project_to_polygon(
                    A, b, vertices, x_true)
                if (x_projected_poly is not None
                        and np.all(np.isfinite(x_projected_poly))):
                    optimality[i, j] = np.sum(
                        (x_true - np.asarray(x_projected_poly)) ** 2)
                    optimality_success[i, j] = True

            if (is_member and truth['radial_success'][i, j]):
                kp = k_max_polytope(A, b, reference, direction)
                if np.isfinite(kp):
                    k_poly[i, j] = kp
                    coverage[i, j] = kp / truth['k_omega'][i, j]

        print(
            f'[evaluate/{tag}] {i + 1}/{len(dthetas)}: '
            f'feas={feasibility_success[i].sum()}/{len(eval_dirs)}, '
            f'opt={optimality_success[i].sum()}/{len(eval_dirs)}, '
            f'rho={np.isfinite(coverage[i]).sum()}/{len(eval_dirs)}')

    evaluation_time = time.time() - start
    out_path = OUT / tag / 'formal_eval.npz'
    np.savez(
        out_path, case=np.asarray(CASENAME), config=np.asarray(tag),
        hidden_sizes=np.asarray(cfg['hidden_sizes']),
        activation=np.asarray(activation_str(cfg)),
        parameter_count=np.asarray(parameter_count), dthetas=dthetas,
        eval_dirs=eval_dirs, predicted_A=predicted_A, predicted_b=predicted_b,
        feasibility=feasibility, optimality=optimality, coverage=coverage,
        k_poly=k_poly, k_omega=truth['k_omega'], x_ref=truth['x_ref'],
        membership_failure=membership_failure,
        polygon_valid=polygon_valid,
        true_support_success=truth['true_support_success'],
        radial_success=truth['radial_success'],
        feasibility_success=feasibility_success,
        optimality_success=optimality_success,
        evaluation_time_s=np.asarray(evaluation_time),
        error_definition=np.asarray('squared_euclidean_distance'),
        coverage_definition=np.asarray('reference_point_radial_kP_over_kOmega'))

    feas_stat = _finite_stats(feasibility[feasibility_success])
    opt_stat = _finite_stats(optimality[optimality_success])
    rho_stat = _finite_stats(coverage)
    under = np.maximum(1.0 - coverage[np.isfinite(coverage)], 0.0)
    over = np.maximum(coverage[np.isfinite(coverage)] - 1.0, 0.0)
    under_stat = _finite_stats(under)
    total = feasibility_success.size
    return {
        'config': tag,
        'hidden_sizes': ','.join(str(x) for x in cfg['hidden_sizes']),
        'activation': activation_str(cfg),
        'parameter_count': parameter_count,
        'n_dtheta': len(dthetas), 'n_directions': len(eval_dirs),
        'feas_mean': feas_stat['mean'], 'feas_median': feas_stat['median'],
        'feas_p95': feas_stat['p95'], 'opt_mean': opt_stat['mean'],
        'opt_median': opt_stat['median'], 'opt_p95': opt_stat['p95'],
        'coverage_mean': rho_stat['mean'],
        'coverage_median': rho_stat['median'],
        'coverage_p05': rho_stat['p05'], 'coverage_p95': rho_stat['p95'],
        'undercoverage_mean': under_stat['mean'],
        'undercoverage_p95': under_stat['p95'],
        'overcoverage_mean': float(np.mean(over)) if over.size else np.nan,
        'membership_failed_cases': int(np.sum(membership_failure)),
        'true_support_success_rate': float(
            np.mean(truth['true_support_success'])),
        'radial_success_rate': float(np.mean(truth['radial_success'])),
        'feas_success_rate': float(np.sum(feasibility_success) / total),
        'opt_success_rate': float(np.sum(optimality_success) / total),
        'phase2_train_time_s': old_time,
    }


def _save_formal_summary(rows):
    with open(OUT / 'formal_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FORMAL_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        OUT / 'formal_summary.npz',
        **{field: np.asarray([row[field] for row in rows])
           for field in FORMAL_SUMMARY_FIELDS})


def run_formal_evaluation():
    """There are only six reviewscase33weights, no retraining."""
    if CASENAME != 'case33bw_ds':
        raise ValueError('R2.3Formal evaluations must usecase33bw_ds.')
    ppc = TD_case.case33bw_ds()
    case_full = TD_case.DScase_train(
        casedata=ppc, model_type='fullnet', plot_flag=False, device=device)
    dthetas = _formal_dthetas(case_full['params']['count'])
    eval_dirs = _formal_directions()
    truth = _prepare_case33_truth(
        case_full['errorcalculator'], ppc,
        case_full['params']['params_dict'], dthetas, eval_dirs)
    old = load_existing_summary()
    rows = []
    for index, cfg in enumerate(CONFIGS, 1):
        tag = config_tag(cfg)
        print(f'\n[evaluate] Configuration {index}/{len(CONFIGS)}: {tag}')
        old_time = float(old.get(tag, {}).get('train_time', np.nan))
        rows.append(_evaluate_existing_config(
            cfg, case_full, ppc, truth, old_time))
    _save_formal_summary(rows)
    print(f'\n[evaluate] case33Formal summary: {OUT / "formal_summary.csv"}')
    from Simulator.draw_pictures.R2_3_network_comparison_plot import main as plot_main
    plot_main()


def main():
    parser = argparse.ArgumentParser(description='R2.3 Network structure×Activation function comparison (PreTrainNet / FullNet / case33formal evaluation) ')
    parser.add_argument('--stage', choices=['pretrain', 'fullnet', 'evaluate'], default=None,
                        help='cover top STAGE; pretrain=training PreTrainNet concurrent disk, fullnet=load weight training FullNet')
    parser.add_argument('--only', default=None,
                        help='cover top ONLY; fullnet The stage only runs the specified config tag (Such as h128-128_act-relu, Comma separated multiple) ')
    args = parser.parse_args()

    stage = args.stage or STAGE
    only = args.only or ONLY
    print(f"running phase: {stage}" + (f"  Just run: {only}" if only else "  Configuration: All"))

    if stage == 'pretrain':
        run_pretrain()
    elif stage == 'evaluate':
        run_formal_evaluation()
    else:
        configs = CONFIGS
        if only:
            wanted = {t.strip() for t in only.split(',')}
            configs = [c for c in CONFIGS if config_tag(c) in wanted]
            print(f"Filter: run only {[config_tag(c) for c in configs]}")
        run_fullnet(configs)


if __name__ == '__main__':
    main()
