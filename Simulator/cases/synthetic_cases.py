# -*- coding: utf-8 -*-
"""Analytical synthetic cases for controlled nonconvex feasible-region experiments."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pyomo.environ import ConcreteModel, Var, Param, Reals
from Simulator.Approximator import ErrorCalculator
from Simulator import PROJECT_ROOT


# ==================== Analytical geometry of the union of two disks (Ω-oracle) ====================
RADIAL_SCAN_STEP = 1e-2
RADIAL_SCAN_MAX_STEPS = 1000
RADIAL_BISECTION_TOL = 1e-8
RADIAL_BISECTION_MAX_ITER = 60


def first_boundary_scan_bisection(x_ref, direction, is_feasible,
                                  scan_step=RADIAL_SCAN_STEP,
                                  scan_max_steps=RADIAL_SCAN_MAX_STEPS,
                                  bisection_tol=RADIAL_BISECTION_TOL,
                                  bisection_max_iter=RADIAL_BISECTION_MAX_ITER):
    """Find the first feasible unit ray/It is impossible to enclose the interval and return the feasible side boundary length."""
    x_ref = np.asarray(x_ref, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if (not np.isfinite(norm) or norm <= 1e-15
            or scan_step <= 0 or scan_max_steps < 1):
        return 0.0
    direction = direction / norm
    if not is_feasible(x_ref):
        return 0.0

    k_feasible = 0.0
    k_infeasible = None
    for index in range(1, scan_max_steps + 1):
        k_trial = index * scan_step
        if not is_feasible(x_ref + k_trial * direction):
            k_infeasible = k_trial
            break
        k_feasible = k_trial
    if k_infeasible is None:
        return 0.0

    for _ in range(bisection_max_iter):
        if k_infeasible - k_feasible <= bisection_tol:
            break
        k_mid = 0.5 * (k_feasible + k_infeasible)
        if is_feasible(x_ref + k_mid * direction):
            k_feasible = k_mid
        else:
            k_infeasible = k_mid
    return float(k_feasible)


def two_disk_centers(d):
    """The centers of two discs of equal diameter (±d,0)."""
    d = float(d)
    return [np.array([-d, 0.0]), np.array([d, 0.0])]


def _disk_k_plus(center, radius, x_ref, direction):
    """When the reference point is inside the disk, the return ray leaves the far root of the disk.."""
    diff = np.asarray(x_ref, dtype=float) - np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    vd = float(direction @ diff)
    dd = float(diff @ diff) - radius * radius
    discriminant = vd * vd - dd
    return -vd + np.sqrt(max(discriminant, 0.0))


def two_disk_k_omega(d, r, x_ref, v):
    """Coverage evaluation uses the largest rayk, with universalcoverageThe script caliber is the same."""
    return max(_disk_k_plus(center, r, x_ref, v)
               for center in two_disk_centers(d))


def two_disk_first_boundary(d, r, x_ref, v):
    """Training detection uses: scan and bisect to getΩ=D₁∪D₂The first radial boundary of."""
    centers = two_disk_centers(d)

    def is_feasible(point):
        return any(np.linalg.norm(point - center) <= r + 1e-12
                   for center in centers)

    return first_boundary_scan_bisection(x_ref, v, is_feasible)


def two_disk_support(d, r, direction):
    """Ω on argmin direction·x (outer normal support point) = Each plate edge -direction Take the farthest point direction·x The smallest one.
    These points all fall on the convex hull - they are sampled by the projection method, and the structure cannot enter the waist.."""
    c = np.asarray(direction, float)
    cn = np.linalg.norm(c)
    if cn < 1e-12:
        return np.zeros(2)
    chat = c / cn
    cands = [ci - r * chat for ci in two_disk_centers(d)]      # Each plate edge -chat the farthest point
    vals = [float(c @ p) for p in cands]
    return cands[int(np.argmin(vals))]


def two_disk_project(d, r, x):
    """point x Arrive Ω=D₁∪D₂ The closest point: in any disk, it returns x; Otherwise, take the closest of the two boundary projections.."""
    x = np.asarray(x, float)
    best, best_d = None, np.inf
    for ci in two_disk_centers(d):
        diff = x - ci
        dist = float(np.linalg.norm(diff))
        if dist <= r:
            return x.copy()                                    # in a certain plate → distance 0
        p = ci + r * diff / dist
        dd = float(np.linalg.norm(x - p))
        if dd < best_d:
            best_d, best = dd, p
    return best


# ==================== Analytical Error Calculator ====================
class TwoDiskEC(ErrorCalculator):
    """Analysis of the Union of Two Disks ErrorCalculator.all"check Ω"All are closed type, skeleton pyomo The model is only used as a cloning vector."""

    def __init__(self, d_init=0.8, r=1.0, A_hat=None, solver='ipopt',
                 x_ref=None, **kwargs):
        self._d = float(d_init)
        self._r = float(r)
        self._x_ref = np.array(x_ref if x_ref is not None else [0.0, 0.0], dtype=float)

        # skeleton model: var_proj + variable scalar d (=θ) .Constraints are empty (the solution is all covered by the following analysis) .
        model = ConcreteModel()
        model.d = Param(initialize=self._d, mutable=True)
        model.var_proj = Var(range(2), domain=Reals, bounds=(-5, 5))
        super().__init__(original_model={'model': model}, A_hat=A_hat,
                         solver=solver, **kwargs)

    # ---- Geometry cache synchronization: d by update_parameters push over ----
    def update_parameters(self, param_dict):
        super().update_parameters(param_dict)                  # base class handle d write in pyomo model (scalar branch)
        if 'd' in param_dict:
            self._d = float(np.asarray(param_dict['d']).reshape(-1)[0])

    # ---- Radial path coverage (radial_probe=True use) ----
    def _compute_x_ref(self, dtheta):
        return self._x_ref.copy()                              # Reference point fixed x̂ (not dependent on dtheta/Trend)

    def _sync_ray_meta(self, dtheta):
        pass                                                   # No power flow element parameters

    def _solve_k_omega(self, x_ref, v):
        return two_disk_first_boundary(self._d, self._r, x_ref, v)

    def radial_dtheta(self, batch_data, i):
        """batch_data['d'][i] → scalar d as"dtheta"pass to calculate_radial_feasopt."""
        return np.asarray(batch_data['d'][i].detach().cpu().numpy()).reshape(-1)

    # ---- Projection path coverage (radial_probe=False use) ; approx Side walking base class (convex polygon gurobi) ----
    def optimize_direction(self, direction, in_approx=False):
        if in_approx:
            return super().optimize_direction(direction, in_approx=True)
        return two_disk_support(self._d, self._r, direction)   # argmin c·x over Union (parse)

    def project(self, target, to_approx=False):
        if to_approx:
            return super().project(target, to_approx=True)
        return two_disk_project(self._d, self._r, target)      # to the nearest point of the union (parsing)


# ==================== case_two_disks: imitation case_ellipse ====================
def case_two_disks(d_init=0.8, r=1.0, total_samples=100, noise_scale=0.1,
                   batch_size=2, model_type='pretrainnet', device='cpu',
                   n_facets=16):
    """Example of union of two disks.θ=d (disc spacing) , PINN learn d→Polygon(A,b) Approximately Ω(d)=D₁∪D₂.
    Default d_init=0.8, r=1.0, Waist coverage at this time ρ≈1.67.
    n_facets: Number of polygon sides (6=Hexagons are the same as case_ellipse; 12/16 Closer to the disc, reducing lack of coverage in rounded areas with small edges) ."""
    dim = 2
    # Uniformly distribute n_facets unit normals; _initialization sets their support values.
    ang = np.linspace(0, 2 * np.pi, n_facets, endpoint=False)
    A_hat = np.column_stack([np.cos(ang), np.sin(ang)])
    errorcalculator = TwoDiskEC(d_init=d_init, r=r, A_hat=A_hat, solver='ipopt')

    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # θ=d perturbation (family) : d ~ d_init + N(0, noise_scale); Geometry consists of update_parameters sync
            return {'d': torch.normal(0, self.noise_scale, (1,), device=device)}

    def callback(error_calculator, epoch):
        len_his = len(error_calculator.training_history['feas'])
        e_feas = np.mean(error_calculator.training_history['feas'][-min(10, len_his):]) if len_his else float('nan')
        e_opt = np.mean(error_calculator.training_history['opt'][-min(10, len_his):]) if len_his else float('nan')
        print(f"  Iter {epoch}: FeasErr={e_feas:.2e}, OptErr={e_opt:.2e}")

    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
            "call_interval": 20,
            "training_callback": callback,
            "optimizer": "sgd",
            "lr": 0.2,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }
    else:
        trainer_configure = {
            "call_interval": 10,
            "training_callback": callback,
            "optimizer": "adam",
            "lr": 0.003,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }

    params = {
        'params_dict': {'d': {'initial_value': np.array([d_init])}},
        'dataloader': DataLoader(CaseData(), batch_size=batch_size, shuffle=True),
        'count': 1,                                            # dim_theta = 1 (only d)
    }
    case_name = 'two_disks'
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'errorcalculator': errorcalculator,
        'params': params,
        'trainer_configure': trainer_configure,
            'result_path': f'{PROJECT_ROOT}/results/ds_proj_revise_V1/synthetic_two_disks/{model_type.lower()}_weights.pth',
        'metadata': {'d_init': d_init, 'r': r, 'dim': dim},
    }


# ==================== Polar star: one connected non-convex domain ====================
# Ω = { x∈ℝ² : |x| ≤ r(θ) },  r(θ) = R(1 + ε cos(kθ)).
# Non-convexity arises from multimodal radial oscillation, rather than a union
# of disconnected sets. Larger epsilon produces sharper peaks and deeper valleys.
# non-convex threshold ε > 1/(1+k²) (k=4 time≈0.059) .Strictly star from origin (r≥R(1−ε)>0) → Well defined radial function.
# θ = ε (non-convex strength) , PINN learn ε → Polygon (A,b), i.e. star-shaped[family].
def star_radius(R, eps, k, theta):
    """Star polar radius r(θ)=R(1+ε cos kθ)."""
    return R * (1.0 + eps * np.cos(k * theta))


def star_k_omega(R, eps, k, x_ref, v):
    """Coverage evaluation uses the largest rayk, with universalcoverageThe script caliber is the same."""
    v = np.asarray(v, dtype=float)
    theta = float(np.arctan2(v[1], v[0]))
    return float(star_radius(R, eps, k, theta))


def star_first_boundary(R, eps, k, x_ref, v):
    """Training detection uses: Scan from the origin and bisect to get the first radial boundary of the star domain."""
    x_ref = np.asarray(x_ref, dtype=float)

    def is_feasible(point):
        theta = float(np.arctan2(point[1], point[0]))
        return np.linalg.norm(point) <= star_radius(R, eps, k, theta) + 1e-12

    return first_boundary_scan_bisection(x_ref, v, is_feasible)


def star_support(R, eps, k, direction):
    """Ω on argmin direction·x (outer normal support point). border x(θ)=r(θ)(cosθ,sinθ),
    target f(θ)=r(θ)·(d·ê_θ) for θ scalar function, no simple closed form → 1D dense mesh min.
    These points all fall on the convex hull - they are sampled by the projection method, and the inter-lobe depression is structurally invisible.."""
    c = np.asarray(direction, float)
    if np.linalg.norm(c) < 1e-12:
        return np.zeros(2)
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    r = star_radius(R, eps, k, theta)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    vals = pts @ c
    return pts[int(np.argmin(vals))].copy()


def star_project(R, eps, k, x):
    """point x Arrive Ω The closest point: at Ω within (|x|≤r(θ_x)) then return x; Otherwise boundary 1D Grid takes closest."""
    x = np.asarray(x, float)
    tx = float(np.arctan2(x[1], x[0]))
    if np.linalg.norm(x) <= star_radius(R, eps, k, tx) + 1e-12:
        return x.copy()
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    r = star_radius(R, eps, k, theta)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    d2 = np.sum((pts - x) ** 2, axis=1)
    return pts[int(np.argmin(d2))].copy()


# ==================== Star Analysis Error Calculator ====================
class StarEC(ErrorCalculator):
    """polar star Ω={|x|≤R(1+ε cos kθ)} analysis ErrorCalculator.
    non-convex source=Single domain boundary radius multi-peak oscillation (different from the union of two disks"Union type"non-convex).
    all"check Ω"All are closed/1D numerical value, skeleton pyomo The model is only used as a cloning vector."""

    def __init__(self, eps_init=0.25, R=1.0, k=4, A_hat=None, solver='ipopt',
                 x_ref=None, **kwargs):
        self._eps = float(eps_init)
        self._R = float(R)
        self._k = int(k)
        self._x_ref = np.array(x_ref if x_ref is not None else [0.0, 0.0], dtype=float)

        # skeleton model: var_proj + variable scalar eps (=θ) .Constraints are empty (the solution is all covered by the following analysis) .
        model = ConcreteModel()
        model.eps = Param(initialize=self._eps, mutable=True)
        model.var_proj = Var(range(2), domain=Reals, bounds=(-5, 5))
        super().__init__(original_model={'model': model}, A_hat=A_hat,
                         solver=solver, **kwargs)

    # ---- Geometry cache synchronization: eps by update_parameters push over ----
    def update_parameters(self, param_dict):
        super().update_parameters(param_dict)                  # base class handle eps write in pyomo model (scalar branch)
        if 'eps' in param_dict:
            self._eps = float(np.asarray(param_dict['eps']).reshape(-1)[0])

    # ---- Radial path coverage (radial_probe=True use) ----
    def _compute_x_ref(self, dtheta):
        return self._x_ref.copy()                              # Reference point fixed origin (star center)

    def _sync_ray_meta(self, dtheta):
        pass                                                   # No power flow element parameters

    def _solve_k_omega(self, x_ref, v):
        return star_first_boundary(
            self._R, self._eps, self._k, x_ref, v)

    def radial_dtheta(self, batch_data, i):
        """batch_data['eps'][i] → scalar ε make"dtheta"pass to calculate_radial_feasopt."""
        return np.asarray(batch_data['eps'][i].detach().cpu().numpy()).reshape(-1)

    # ---- Projection path coverage (radial_probe=False use) ; approx Side walking base class (convex polygon gurobi) ----
    def optimize_direction(self, direction, in_approx=False):
        if in_approx:
            return super().optimize_direction(direction, in_approx=True)
        return star_support(self._R, self._eps, self._k, direction)

    def project(self, target, to_approx=False):
        if to_approx:
            return super().project(target, to_approx=True)
        return star_project(self._R, self._eps, self._k, target)


# ==================== case_star: imitation case_two_disks ====================
def case_star(eps_init=0.25, R=1.0, k=4, total_samples=100, noise_scale=0.05,
              batch_size=2, model_type='pretrainnet', device='cpu', n_facets=16):
    """Polar coordinate star calculation example.θ=ε (non-convex strength) , PINN learn ε→Polygon(A,b) Approximately Ω(ε)={|x|≤R(1+ε cos kθ)}.
    eps_init=0.25,R=1,k=4: Four-petaled flower with valley between petals ρ>1 (non-convex), valve cusp ρ≈1.
    n_facets: Number of polygon sides (16 Used; the cusp may be under-covered, which is ρ revealed) ."""
    dim = 2
    # Uniformly distribute n_facets unit normals; _initialization sets their support values.
    ang = np.linspace(0, 2 * np.pi, n_facets, endpoint=False)
    A_hat = np.column_stack([np.cos(ang), np.sin(ang)])
    errorcalculator = StarEC(eps_init=eps_init, R=R, k=k, A_hat=A_hat, solver='ipopt')

    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # θ=ε perturbation (family) : eps ~ eps_init + N(0, noise_scale); r≥R(1−ε)>0 request ε<1
            return {'eps': torch.normal(0, self.noise_scale, (1,), device=device)}

    def callback(error_calculator, epoch):
        len_his = len(error_calculator.training_history['feas'])
        e_feas = np.mean(error_calculator.training_history['feas'][-min(10, len_his):]) if len_his else float('nan')
        e_opt = np.mean(error_calculator.training_history['opt'][-min(10, len_his):]) if len_his else float('nan')
        print(f"  Iter {epoch}: FeasErr={e_feas:.2e}, OptErr={e_opt:.2e}")

    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
            "call_interval": 20,
            "training_callback": callback,
            "optimizer": "sgd",
            "lr": 0.2,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }
    else:
        trainer_configure = {
            "call_interval": 10,
            "training_callback": callback,
            "optimizer": "adam",
            "lr": 0.003,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }

    params = {
        'params_dict': {'eps': {'initial_value': np.array([eps_init])}},
        'dataloader': DataLoader(CaseData(), batch_size=batch_size, shuffle=True),
        'count': 1,                                            # dim_theta = 1 (only ε)
    }
    case_name = 'star'
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'errorcalculator': errorcalculator,
        'params': params,
        'trainer_configure': trainer_configure,
            'result_path': f'{PROJECT_ROOT}/results/ds_proj_revise_V1/synthetic_star/{model_type.lower()}_weights.pth',
        'metadata': {'eps_init': eps_init, 'R': R, 'k': k, 'dim': dim},
    }
