# -*- coding: utf-8 -*-
"""Run the controlled two-disk-union nonconvex feasible-region experiment."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import csv
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as PatchCircle

from Simulator.Approximator import PreTrainNet, FullNet, compute_loss, Trainer
from Simulator.cases.synthetic_cases import (
    RADIAL_BISECTION_TOL,
    RADIAL_SCAN_STEP,
    case_two_disks,
    two_disk_centers,
    two_disk_k_omega,
)
from Simulator import PROJECT_ROOT
from Simulator.reference_point import (
    REFERENCE_MEMBERSHIP_TOL,
    reference_point_membership,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ Configuration ============
D_INIT = 0.8       # disc spacing d; with R=1.0 Waist coverage when paired ρ≈1.67
R = 1.0            # Disk radius
N_PRE = 150        # PreTrainNet Iterate
N_FULL_PH1 = 50    # FullNet phase1 iteration (synthesis: rate_opt_feas=PH1_RATE)
N_FULL_PH2 = 60    # FullNet phase2 Iteration (focus on feasibility: rate_opt_feas=PH2_RATE)
PH1_RATE = 0.6     # phase1 rate_opt_feas (feas+opt Comprehensive)
PH1_LR = 1e-3      # The default learning rate shared by both methods; no learning rate sensitivity analysis was performed
PH2_RATE = 1e-4    # phase2 rate_opt_feas (feas Overwhelm; align main_ds.py / R2.5 runner)
PH2_LR = 1e-5      # Default second-stage learning rate shared by both methods; no sensitivity analysis performed
N_DIRS_PROJECTION = 8  # Projection and radial methods use the same number of search directions for each sample
N_DIRS_RADIAL = N_DIRS_PROJECTION
N_DIRS_EVAL = 120  # Assessment ρ The number of directions
N_SNAP = 6         # Number of pre-training feasible region evolution snapshots
N_FACETS = 16      # Number of polygon sides (6 too little→Lack of coverage in round areas; 16 More attached to the two discs)
OUT = PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'synthetic_two_disks'
PRETRAIN_CACHE = OUT / f'pretrain_warmstart_nfacets{N_FACETS}.npz'
SKIP_PRETRAIN = False  # False: Re-pretrain every run and overwrite updates warm-start cache
# Fairness Caliber: Two Laws Sharing the Network, warm-start, Learning rate and number of search directions per sample.
# The supervised constructions use different query counts; oracle_calls records the observed total.
# This conventional learning rate is not presented as the result of a sensitivity study.


# ============ Tools ============
def k_max_polytope(A, b, x_ref, v):
    """Polyhedron Ax≤b from x_ref along unit direction v The radial maximum k (closed type, numpy) ."""
    is_member, _ = reference_point_membership(
        A, b, x_ref, REFERENCE_MEMBERSHIP_TOL)
    if not is_member:
        return np.nan
    Av = A @ v
    Ax = A @ x_ref
    ratio = np.where(Av > 1e-9, (b - Ax) / np.where(Av > 1e-9, Av, 1.0), np.inf)
    k = ratio.min()
    return k if np.isfinite(k) else 0.0


def gen_dirs_even(n):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(th), np.sin(th)))


def get_polygon(model, d):
    """take d polygon at (A,b).FullNet The input is delta=d-d_init: during training _combine_batch Feeding is noise,
    frame handle initial_value+noise write to EC → What the model learns is delta→polygon(d_init+delta).So query d Must feed d-d_init."""
    delta = float(d) - D_INIT
    with torch.no_grad():
        A, b = model(torch.tensor([delta], dtype=torch.float32, device=device))
    return A[0].detach().cpu().numpy(), b[0].detach().cpu().numpy()


# ============ stage 1: PreTrainNet warm-start ============
def run_pretrain():
    case = case_two_disks(d_init=D_INIT, r=R, model_type='pretrainnet', device=device,
                          n_facets=N_FACETS)
    model = PreTrainNet(case['A_hat'], case['b_hat'], device=device).to(device)
    trainer = Trainer(model=model, error_calculator=case['errorcalculator'],
                      compute_loss=compute_loss)
    trainer.configure(**case['trainer_configure'])
    trainer.initialize()
    print("=== 1) PreTrainNet warm-start ===")
    trainer.train(n_train=N_PRE*10, params_data=case['params'])
    A_out, b_out = model()
    return A_out[0].detach().cpu().numpy(), b_out[0].detach().cpu().numpy()


# ============ stage 1': PreTrainNet pre-training + Feasible region evolution snapshot (only look at pre-training) ============
def run_pretrain_snapshots():
    """Only run pre-training and catch up along the training process N_SNAP Take a snapshot of the feasible region for drawing an evolution diagram.
    Return (snapshots, A_pre, b_pre): snapshots=[(iter, A, b), ...]; A_pre/b_pre for subsequent stages.

    Implementation: Replace with closure callback case Comes with it training_callback (closure capture model with snapshots list),
    Grabs a snapshot of the current polygon when hitting the snapshot target iteration point. The first one is the true initial tangent polygon before training.."""
    case = case_two_disks(d_init=D_INIT, r=R, model_type='pretrainnet', device=device,
                          n_facets=N_FACETS)
    model = PreTrainNet(case['A_hat'], case['b_hat'], device=device).to(device)
    trainer = Trainer(model=model, error_calculator=case['errorcalculator'],
                      compute_loss=compute_loss)
    trainer.configure(**case['trainer_configure'])

    total_iter = N_PRE * 10
    call_interval = trainer.params.get('call_interval', 20)
    # The first snapshot is the initial tangent polygon; later snapshots follow call_interval.
    snapshots = []
    with torch.no_grad():
        A0, b0 = model()
    snapshots.append((0, A0[0].detach().cpu().numpy().copy(),
                      b0[0].detach().cpu().numpy().copy()))
    rest = [int(np.ceil(t / call_interval) * call_interval)
            for t in np.linspace(total_iter // (N_SNAP - 1), total_iter, N_SNAP - 1, dtype=int)]
    targets = [0] + rest                                  # targets[0]=0 Manually captured; callback from targets[1] match

    def snap_callback(ec, i_iter):
        # Use the original callback: Print recent FeasErr/OptErr
        len_his = len(ec.training_history['feas'])
        e_feas = np.mean(ec.training_history['feas'][-min(10, len_his):]) if len_his else float('nan')
        e_opt = np.mean(ec.training_history['opt'][-min(10, len_his):]) if len_his else float('nan')
        # Hit the next snapshot target to grab one (i_iter Already call_interval multiples of)
        nxt = len(snapshots)
        if nxt < len(targets) and i_iter >= targets[nxt]:
            with torch.no_grad():
                A_s, b_s = model()
            snapshots.append((int(i_iter),
                              A_s[0].detach().cpu().numpy().copy(),
                              b_s[0].detach().cpu().numpy().copy()))
            print(f"    >> Snapshot {len(snapshots)}/{N_SNAP} @ iter {i_iter}")
        print(f"  Iter {i_iter}: FeasErr={e_feas:.2e}, OptErr={e_opt:.2e}")

    trainer.params['training_callback'] = snap_callback   # Closure callback replacement case Bring your own callback
    trainer.initialize()
    print("=== pre-training PreTrainNet (Capture snapshots of feasible domain evolution) ===")
    trainer.train(n_train=total_iter, params_data=case['params'])

    # Finishing: Make sure the last picture is the final state of training
    if snapshots[-1][0] < targets[-1]:
        with torch.no_grad():
            A_s, b_s = model()
        snapshots.append((int(total_iter),
                          A_s[0].detach().cpu().numpy().copy(),
                          b_s[0].detach().cpu().numpy().copy()))

    A_pre, b_pre = snapshots[-1][1], snapshots[-1][2]
    return snapshots, A_pre, b_pre


# ============ Draw feasible region evolution diagram (general: pretrain / projection / radial share) ============
def plot_evolution(name, snapshots, color, subtitle):
    """2×3 Grid: each subplot = two discs(trueΩ,red) + Currently learned convex polygon(from x̂ Radial tracking {Ax≤b})."""
    x_ref = np.array([0.0, 0.0])
    dirs = gen_dirs_even(200)
    k_om = np.array([two_disk_k_omega(D_INIT, R, x_ref, v) for v in dirs])
    centers = two_disk_centers(D_INIT)

    n = len(snapshots)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for idx, (it, A, b) in enumerate(snapshots):
        ax = axes[idx]
        for ci in centers:                                # true Ω (Two discs filled)
            ax.add_patch(PatchCircle(ci, R, facecolor='#C44E52', alpha=0.13, edgecolor='none'))
        bnd = x_ref + k_om[:, None] * dirs
        ax.plot(np.append(bnd[:, 0], bnd[0, 0]), np.append(bnd[:, 1], bnd[0, 1]),
                '-', color='#C44E52', lw=1.6, alpha=0.85,
                label=r'true $\Omega$' if idx == 0 else None)
        kP = np.clip(np.array([k_max_polytope(A, b, x_ref, v) for v in dirs]), -5.0, 5.0)
        poly = x_ref + kP[:, None] * dirs
        ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
                '-', color=color, lw=2.0,
                label=f'{name} polygon' if idx == 0 else None)
        ax.plot(*x_ref, 'k*', ms=10)
        ax.set_xlim(-2.1, 2.1); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect('equal'); ax.grid(linestyle='--', alpha=0.3)
        ax.set_title(f'iter = {it}', fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8, loc='lower left')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{subtitle}  (d={D_INIT})', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.mkdir(parents=True, exist_ok=True)
    out_png = OUT / f'{name}_evolution.png'
    out_svg = OUT / f'{name}_evolution.svg'
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_svg)
    print(f"saved {name} Evolution diagram: {out_png}")


# ============ stage 2: training projection / radial FullNet (two stages + Evolutionary Snapshot) ============
def train_version(name, radial_probe, A_pre, b_pre):
    """train one FullNet version, two stages (= main_ds.py / R2.5 runner original manuscript schedule) :
       phase1 Comprehensive(rate_opt_feas=0.6) → phase2 Focus on feasibility(rate_opt_feas=1e-4).
    Synchronous capture N_SNAP Zhang feasible region snapshot (constant d=D_INIT assessment). Return (model, snapshots)."""
    case = case_two_disks(d_init=D_INIT, r=R, model_type='fullnet', device=device,
                          n_facets=N_FACETS)
    ec = case['errorcalculator']
    # The radial method brackets the first infeasible point and then bisects the interval.
    # projectionversionradial_probe=False, Do not trigger this path.
    ec.attach_radial(None, {}, n_dirs=N_DIRS_RADIAL)
    model = FullNet(dim_theta=case['params']['count'], A_init=A_pre, b_init=b_pre,
                    hidden_sizes=[128], activation='relu', device=device).to(device)
    trainer = Trainer(model=model, error_calculator=ec, compute_loss=compute_loss)
    trainer.configure(**case['trainer_configure'])
    n_dirs = N_DIRS_RADIAL if radial_probe else N_DIRS_PROJECTION
    trainer.configure(radial_probe=radial_probe, n_cal=n_dirs)

    # ---- Evolution snapshot: use cb_count (accumulated across two stages) for indexing, because i_iter every time train() will reset ----
    snapshots = []
    A0, b0 = get_polygon(model, D_INIT)                  # debut: warm-start (ZeroInit → model(0)=(A_pre,b_pre))
    snapshots.append((0, A0.copy(), b0.copy()))
    n_batches = len(case['params']['dataloader'])
    call_interval = trainer.params.get('call_interval', 10)
    total_iter = (N_FULL_PH1 + N_FULL_PH2 + 2) * n_batches
    total_cb = max(1, total_iter // call_interval)
    cb_targets = np.linspace(0, total_cb, N_SNAP, dtype=int).tolist()
    cb_count = [0]

    def snap_callback(ecb, i_iter):
        cb_count[0] += 1
        len_his = len(ecb.training_history['feas'])
        e_feas = np.mean(ecb.training_history['feas'][-min(10, len_his):]) if len_his else float('nan')
        e_opt = np.mean(ecb.training_history['opt'][-min(10, len_his):]) if len_his else float('nan')
        nxt = len(snapshots)
        if nxt < len(cb_targets) and cb_count[0] >= cb_targets[nxt]:
            A_s, b_s = get_polygon(model, D_INIT)
            snapshots.append((int(cb_count[0]), A_s.copy(), b_s.copy()))
            print(f"    >> [{name}] Snapshot {len(snapshots)}/{N_SNAP} @ cb {cb_count[0]}")
        print(f"  [{name}] cb={cb_count[0]}: FeasErr={e_feas:.2e}, OptErr={e_opt:.2e}")

    trainer.params['training_callback'] = snap_callback

    # ---- Phase1: Comprehensive (feas+opt, rate_opt_feas=PH1_RATE; lr=PH1_LR, Used from the original fullnet Configuration 3e-3) ----
    trainer.configure(rate_opt_feas=PH1_RATE, lr=PH1_LR)
    trainer.initialize()
    print(f"\n=== training {name} version phase1 (radial_probe={radial_probe}, rate={PH1_RATE}, lr={PH1_LR}) ===")
    phase1_start = time.perf_counter()
    trainer.train(n_train=N_FULL_PH1, params_data=case['params'])
    phase1_time = time.perf_counter() - phase1_start

    # ---- Phase2: Focus on feasibility (rate=PH2_RATE, lr=PH2_LR; Both methods use the same preset) ----
    # No learning-rate sensitivity study is performed; report only the configured value.
    trainer.configure(rate_opt_feas=PH2_RATE, lr=PH2_LR)
    trainer.initialize()
    print(f"=== training {name} version phase2 (rate={PH2_RATE}, lr={PH2_LR}) ===")
    phase2_start = time.perf_counter()
    trainer.train(n_train=N_FULL_PH2, params_data=case['params'])
    phase2_time = time.perf_counter() - phase2_start

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / f'{name}_weights.pth')

    if len(snapshots) < N_SNAP:                           # Closing: Make sure the last one=End of training
        A_s, b_s = get_polygon(model, D_INIT)
        snapshots.append((int(cb_count[0]), A_s.copy(), b_s.copy()))
    run_info = {
        'method': name,
        'phase1_time': float(phase1_time),
        'phase2_time': float(phase2_time),
        'total_train_time': float(phase1_time + phase2_time),
        'directions_per_sample': int(n_dirs),
        **{k: int(v) for k, v in trainer.probe_statistics.items()},
    }
    return model, snapshots, run_info


# ============ stage 3: Assessment ρ + Draw a picture ============
def calculate_rhos(models):
    x_ref = np.array([0.0, 0.0])
    dirs = gen_dirs_even(N_DIRS_EVAL)
    k_om = np.array([two_disk_k_omega(D_INIT, R, x_ref, v) for v in dirs])

    polys, rhos = {}, {}
    for name, m in models.items():
        A, b = get_polygon(m, D_INIT)
        kP = np.array([k_max_polytope(A, b, x_ref, v) for v in dirs])
        polys[name] = kP
        rhos[name] = kP / np.maximum(k_om, 1e-12)
    return dirs, k_om, polys, rhos


def plot_and_eval(models):
    x_ref = np.array([0.0, 0.0])
    dirs, k_om, polys, rhos = calculate_rhos(models)

    # ---- Drawing: left=Geometry, right=ρ(θ) polar coordinates ----
    fig = plt.figure(figsize=(14, 6))
    axL = fig.add_subplot(1, 2, 1)
    axR = fig.add_subplot(1, 2, 2, projection='polar')
    colors = {'projection': '#4C72B0', 'radial': '#55A868'}

    # true Ω (Two discs filled) + true radial boundary
    for ci in two_disk_centers(D_INIT):
        axL.add_patch(PatchCircle(ci, R, facecolor='#C44E52', alpha=0.15, edgecolor='none'))
    bnd = x_ref + k_om[:, None] * dirs
    axL.plot(np.append(bnd[:, 0], bnd[0, 0]), np.append(bnd[:, 1], bnd[0, 1]),
             '-', color='#C44E52', lw=2.2, label=r'true $\Omega=D_1\cup D_2$')
    # Polygons learned in both versions (from x̂ Radial tracking boundaries)
    for name, kP in polys.items():
        poly = x_ref + kP[:, None] * dirs
        axL.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
                 '-', color=colors[name], lw=1.8, label=f'{name} polygon')
    axL.plot(*x_ref, 'k*', ms=13, label=r'reference $\hat{x}$')
    axL.set_xlim(-2.1, 2.1); axL.set_ylim(-1.6, 1.6)
    axL.set_aspect('equal'); axL.grid(linestyle='--', alpha=0.3)
    axL.set_xlabel(r'$x_1$'); axL.set_ylabel(r'$x_2$')
    axL.set_title(f'(a) Learned polygons vs true non-convex $\\Omega$ (d={D_INIT})', fontsize=10)
    axL.legend(fontsize=8, loc='lower left')

    # ρ(θ)
    theta = np.arctan2(dirs[:, 1], dirs[:, 0])
    axR.plot(np.append(theta, theta[0]), np.append(np.ones(len(theta)), 1.0),
             'k--', lw=1, alpha=0.6, label='ρ=1 (exact cover)')
    for name in models:
        rho = rhos[name]
        axR.plot(np.append(theta, theta[0]), np.append(rho, rho[0]),
                 '-', color=colors[name], lw=2, label=f'{name} ρ')
    rho_max = max(rhos['radial'].max(), rhos['projection'].max())
    axR.set_rmin(0); axR.set_rmax(rho_max * 1.12)
    axR.set_title('(b) Reference-point coverage ρ (waist → ρ>1 over-cover)', fontsize=10, pad=14)
    axR.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.13))
    axR.grid(linestyle='--', alpha=0.3)

    fig.suptitle('Controlled non-convex two-disk Ω: projection probing vs '
                 'reference-point radial probing', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = OUT / 'two_disks_radial_vs_projection.png'
    out_svg = OUT / 'two_disks_radial_vs_projection.svg'
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_svg)

    # ---- Summary ----
    print("\n=== ρ Summary (d={:.2f}, true Ω Waist ρ_convex hull≈{:.2f}) ===".format(
        D_INIT, 1.0 / np.sqrt(max(1e-9, 1 - D_INIT ** 2))))
    for name in models:
        rho = rhos[name]
        print(f"  {name:<11}: ρ mean={np.mean(rho):.3f}  median={np.median(rho):.3f}  "
              f"max={np.max(rho):.3f}  (>1 Indicates over coverage)")
    print(f"\nsaved: {out_png}\n        {out_svg}")
    return rhos


def save_results(run_rows, rhos):
    """Save the coverage, training time, effective supervision number and query budget of a single comparison."""
    fields = [
        'method', 'rho_mean', 'rho_median', 'rho_min', 'rho_max',
        'mean_abs_rho_error', 'mean_overcoverage', 'mean_undercoverage',
        'overcoverage_direction_rate', 'undercoverage_direction_rate',
        'phase1_time', 'phase2_time', 'total_train_time',
        'optimizer_steps', 'training_samples', 'effective_training_samples',
        'direction_probes',
        'true_domain_oracle_calls', 'polytope_oracle_calls',
        'feas_valid_samples', 'opt_valid_samples', 'valid_supervision_records',
        'coverage_rate', 'area_ratio', 'overcoverage_area_rate',
        'undercoverage_area_rate',
        'directions_per_sample', 'phase1_lr', 'phase2_lr',
        'phase1_rate', 'phase2_rate',
    ]
    with (OUT / 'comparison_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(run_rows)

    methods = ('projection', 'radial')
    rho_array = np.stack([rhos[method] for method in methods])
    np.savez_compressed(
        OUT / 'coverage_raw.npz',
        methods=np.asarray(methods), directions=gen_dirs_even(N_DIRS_EVAL),
        rho=rho_array,
        definition=np.array(
            'rho=k_P/k_Omega; coverage k_Omega is the maximum nonnegative '
            'ray length (max-k), consistent with approximate_polygon_coverage.py; '
            'rho>1 overcoverage; rho<1 undercoverage'),
        radial_training_boundary_definition=np.array(
            'first boundary found by outward scan '
            f'(step={RADIAL_SCAN_STEP}) and feasible-side bisection '
            f'(tol={RADIAL_BISECTION_TOL})'))


def add_coverage_metrics(run_info, rho, k_omega):
    """directionρwith area coverage indicators.

    Both the synthetic true domain and the predicted convex polygon are star-shaped with respect to the reference point, so the polar coordinate area is used
    0.5*∫r(θ)^2 dθ Calculate intersection, over-coverage and under-coverage ratios.
    """
    tol = 1e-6
    over = np.maximum(rho - 1.0, 0.0)
    under = np.maximum(1.0 - rho, 0.0)
    k_poly = rho * k_omega
    area_true = float(np.mean(k_omega ** 2))
    area_poly = float(np.mean(k_poly ** 2))
    area_intersection = float(np.mean(np.minimum(k_omega, k_poly) ** 2))
    return {
        **run_info,
        'rho_mean': float(rho.mean()), 'rho_median': float(np.median(rho)),
        'rho_min': float(rho.min()), 'rho_max': float(rho.max()),
        'mean_abs_rho_error': float(np.mean(np.abs(rho - 1.0))),
        'mean_overcoverage': float(over.mean()),
        'mean_undercoverage': float(under.mean()),
        'overcoverage_direction_rate': float(np.mean(rho > 1.0 + tol)),
        'undercoverage_direction_rate': float(np.mean(rho < 1.0 - tol)),
        'coverage_rate': area_intersection / max(area_true, 1e-12),
        'area_ratio': area_poly / max(area_true, 1e-12),
        'overcoverage_area_rate': ((area_poly - area_intersection)
                                   / max(area_true, 1e-12)),
        'undercoverage_area_rate': ((area_true - area_intersection)
                                    / max(area_true, 1e-12)),
        'phase1_lr': PH1_LR, 'phase2_lr': PH2_LR,
        'phase1_rate': PH1_RATE, 'phase2_rate': PH2_RATE,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='R2.5Non-convex domain of double disk union: comparison between projective supervision and radial supervision')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Allow overwriting existingcomparison_metrics.csvand supporting graphs and arrays')
    parser.add_argument(
        '--reuse-pretrain', action='store_true', default=SKIP_PRETRAIN,
        help='Ifwarm-startIf the cache exists, reuse it and skip it.PreTrainNettraining')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Optional random seed; defaultNoneto retain the non-fixed randomness of the original experiment')
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    result_marker = OUT / 'comparison_metrics.csv'
    if result_marker.exists() and not args.overwrite:
        raise FileExistsError(
            f'Result already exists: {result_marker}\n'
            'To retrain and override, add --overwrite.')
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # 1) pre-training warm-start (Reusable cache skip) + evolution
    if args.reuse_pretrain and PRETRAIN_CACHE.exists():
        d_cache = np.load(PRETRAIN_CACHE)
        A_pre, b_pre = d_cache['A_pre'], d_cache['b_pre']
        print(f"[main] Reuse pre-training warm-start: {PRETRAIN_CACHE}  (--reuse-pretrain)")
    else:
        pre_snaps, A_pre, b_pre = run_pretrain_snapshots()
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(PRETRAIN_CACHE, A_pre=A_pre, b_pre=b_pre)
        print(f"[main] Pre-training completed, warm-start Already saved: {PRETRAIN_CACHE}")
        plot_evolution('pretrain', pre_snaps, '#4C72B0',
                       'PreTrainNet warm-start: polygon converging onto the convex hull')

    # 2) Single comparison training; no fixed random seed is set.
    model_proj, proj_snaps, proj_info = train_version(
        'projection', False, A_pre, b_pre)
    model_rad, rad_snaps, rad_info = train_version(
        'radial', True, A_pre, b_pre)
    models = {'projection': model_proj, 'radial': model_rad}
    _, k_omega, _, rhos = calculate_rhos(models)
    run_rows = [
        add_coverage_metrics(proj_info, rhos['projection'], k_omega),
        add_coverage_metrics(rad_info, rhos['radial'], k_omega),
    ]
    plot_evolution('projection', proj_snaps, '#4C72B0',
                   'projection FullNet (support-point probe → structurally sees convex hull)')
    plot_evolution('radial', rad_snaps, '#55A868',
                   'radial FullNet (reference-point probe → convex approximation of non-convex Ω)')
    plot_and_eval(models)
    save_results(run_rows, rhos)
    print(f"[main] The original coverage and statistics of a single comparison have been saved to: {OUT}")


if __name__ == '__main__':
    main()
