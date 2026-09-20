# -*- coding: utf-8 -*-
"""Neural feasible-region approximation, training, and error-evaluation utilities."""
import copy

from pyomo.environ import *
from pyomo.common.errors import ApplicationError
import numpy as np
# import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import time

from Simulator.reference_point import (
    REFERENCE_MEMBERSHIP_TOL,
    ReferencePointCalculator,
    reference_point_membership,
)


class ErrorCalculator():
    def __init__(self,
                 original_model: Dict,
                 A_hat,
                 b_hat = None,
                 is_epigraph = False,
                 solver: str = 'gurobi',):
        """
        Args:
            original_model: A dictionary containing the following keys
                - 'variables': variable dictionary {variable name: cvxpyvariable}
                - 'constraints': constraint list
                - 'agg_var_name': 'P_AGG'
            solver: Optimization solver
        """
        self.is_epigraph = is_epigraph
        self.baseline = original_model.get('baseline')
        self.original_model_dict = original_model
        self.original_model = original_model['model'].clone()
        self.dim = len(self.original_model.var_proj)# Dimensions
        self.original_model.x_apx = Param(range(self.dim),initialize=0, mutable=True)
        self.original_model.direction = Param(range(self.dim),initialize=0, mutable=True)
        self.original_model.min_direction = (  #Xorg(V)=argmin V*Xorg
            Objective(expr=sum(self.original_model.direction[j] * self.original_model.var_proj[j]
                               for j in range(self.dim)),
                      sense=minimize))
        self.original_model.min_error = (
            Objective(expr=sum((self.original_model.var_proj[j] - self.original_model.x_apx[j])**2
                               for j in range(self.dim)),
                      sense=minimize))
        if is_epigraph:
            self.original_model.max_obj = Objective(expr = self.original_model.obj, sense = maximize)
            self.original_model.max_obj.deactivate()
        self.solver = SolverFactory(solver,tee = False)  #tee Parameters: Controls whether the output of the solver is printed to the console
        # self.solver.options['OutputFlag'] = 0

        # self.solver.options['constr_viol_tol'] = 1e-6

        self.solver_str = solver

        self.approx_model = ConcreteModel(name='Approximator')
        self.approx_model.var_proj = Var(range(self.dim), initialize=0.0)
        self.approx_model.x_org = Param(range(self.dim),initialize=0, mutable=True)
        self.approx_model.direction = Param(range(self.dim), initialize=0, mutable=True)
        self.approx_model.min_direction = (  #Xapx(V)=argmin V*Xapx
            Objective(expr=sum(self.approx_model.direction[j] * self.approx_model.var_proj[j]
                               for j in range(self.dim)),
                      sense=minimize))
        self.approx_model.min_error = (
            Objective(expr=sum((self.approx_model.var_proj[j] - self.approx_model.x_org[j]) ** 2
                               for j in range(self.dim)),
                      sense=minimize))
        # Initialize approximate polyhedron
        n_cons = A_hat.shape[0]  #OK
        self.approx_model.A = Param(range(n_cons), range(self.dim), mutable=True)
        self.approx_model.b = Param(range(n_cons), mutable=True)
        self._initialization(A_hat, b_hat)

        def matrix_constraint_rule(model, i):
            return sum(model.A[i,j] * model.var_proj[j]
                       for j in range(self.dim)) <= model.b[i]
        self.approx_model.constraints = Constraint(range(n_cons), rule=matrix_constraint_rule)

        FR_params = original_model.get('FR_params', None)
        if is_epigraph and (FR_params is not None) and (FR_params['A_fr'] is not None) and (FR_params['b_fr'] is not None):
            A_fr = FR_params['A_fr']
            b_fr = FR_params['b_fr']
            def epigraph_fr_rule(model, i):
                return sum(A_fr[i, j] * model.var_proj[j]
                           for j in range(self.dim-1)) <= b_fr[i]
            self.approx_model.epigraph_fr = Constraint(range(A_fr.shape[0]), rule=epigraph_fr_rule)
        self.cvx_solver = SolverFactory('gurobi',tee = False)

        self.params = {'active_tol': 1e-5}
        self.training_history = {
            'feas': [],
            'opt': []
        }
        self._iter = 0
    def _initialization(self, A_hat,b_hat = None):
        """Initialize approximate polyhedral structure"""
        if self.is_epigraph:
            self._cal_max_obj()
            self.original_model.fmax = Param(initialize=self.fmax,mutable=True)
            self.original_model.f_ub = Constraint(expr=self.original_model.var_proj[self.dim-1]<=self.original_model.fmax)
        self.A_hat = A_hat
        n_constraints = self.A_hat.shape[0]
        for i in range(n_constraints):
            for j in range(self.dim):
                self.approx_model.A[i, j].value = self.A_hat[i, j]
        if b_hat is None:
            self.b_hat = np.zeros(n_constraints)  #Creates an array of specified shape and data type, filled with zeros.
            for i in range(n_constraints):
                self.b_hat[i] = self.A_hat[i] @ self.optimize_direction(-self.A_hat[i])
                self.approx_model.b[i].value = self.b_hat[i]
        else:
            self.b_hat = b_hat
            for i in range(n_constraints):
                self.approx_model.b[i].value = self.b_hat[i]


    def _cal_max_obj(self):
        self.original_model.min_direction.deactivate()
        self.original_model.min_error.deactivate()
        self.original_model.max_obj.activate()
        if hasattr(self.original_model, 'fmax'):
            self.original_model.fmax.set_value(np.inf)
        solver = SolverFactory(self.original_model_dict.get('fmax_solver',self.solver_str))
        results = solver.solve(self.original_model)
        if results.solver.termination_condition == TerminationCondition.optimal:
            self.fmax = value(self.original_model.obj)
        self.original_model.max_obj.deactivate()

    def configure(self, **kwargs):
        """Update configuration parameters."""
        self.params.update(kwargs)

    def project(self, target: np.ndarray, to_approx: bool = False) -> Optional[np.ndarray]:  # Optional[] is a type hint (type hint) part, indicating that the function may return a certain type of value or may return None.
        if to_approx:
            self.approx_model.min_direction.deactivate()
            self.approx_model.min_error.activate()
            for j in range(self.dim):
                self.approx_model.x_org[j].value = target[j]
            try:
                results = self.cvx_solver.solve(self.approx_model)
                if results.solver.termination_condition == TerminationCondition.optimal:
                    return np.array([self.approx_model.var_proj[j].value for j in range(self.dim)])
                print(('project',to_approx,results.solver.termination_condition,target))
            except ApplicationError:
                raise
            except Exception as exc:
                print(('project solver exception', to_approx, repr(exc),
                       'iteration', getattr(self, '_iter', 'initialization')))
                return None
        else:
            self.original_model.min_direction.deactivate()
            self.original_model.min_error.activate()
            for j in range(self.dim):
                self.original_model.x_apx[j].value = target[j]
            try:
                results = self.solver.solve(self.original_model)
                if results.solver.termination_condition == TerminationCondition.optimal:
                    return np.array([self.original_model.var_proj[j].value for j in range(self.dim)])
                # else:
                #     print(1)
                print(('project',to_approx,results.solver.termination_condition,target))
            except ApplicationError:
                raise
            except Exception as exc:
                print(('project solver exception', to_approx, repr(exc),
                       'iteration', getattr(self, '_iter', 'initialization')))
                return None
        return None
    def optimize_direction(self,
                          direction: np.ndarray,
                          in_approx: bool = False) -> Optional[np.ndarray]:
        if in_approx:
            self.approx_model.min_error.deactivate()  # Constraint failure
            self.approx_model.min_direction.activate()  # Reactivate constraints
            for j in range(self.dim):
                self.approx_model.direction[j].value = direction[j]
            # print((self.A_hat,self.b_hat))
            try:
                results = self.cvx_solver.solve(self.approx_model)
                if results.solver.termination_condition == TerminationCondition.optimal:
                    return np.array([self.approx_model.var_proj[j].value for j in range(self.dim)])
                print(('direction', in_approx, results.solver.termination_condition, direction))
            except ApplicationError:
                raise
            except Exception as exc:
                print(('direction solver exception', in_approx, repr(exc),
                       'iteration', getattr(self, '_iter', 'initialization')))
                return None
        else:
            self.original_model.min_error.deactivate()
            self.original_model.min_direction.activate()
            for j in range(self.dim):
                self.original_model.direction[j].value = direction[j]
            try:  #in try Blocks that place code that may cause errors
                results = self.solver.solve(self.original_model)
                if results.solver.termination_condition == TerminationCondition.optimal:
                    return np.array([self.original_model.var_proj[j].value for j in range(self.dim)])

                print(('direction', in_approx, results.solver.termination_condition, direction))
            except ApplicationError:
                raise
            except Exception as exc:  # Non-solver startup faults will still be returned according to the original logic.None.
                print(('direction solver exception', in_approx, repr(exc),
                       'iteration', getattr(self, '_iter', 'initialization')))
                return None
        return None
    def _find_active(self,x_apx):
        # 1. Find activation constraints
        residuals = self.A_hat @ x_apx - self.b_hat  #@: Matrix multiplication
        active_indices = np.where(np.abs(residuals) < self.params['active_tol'])[0]  #np.absWhat is returned is a new array containing the absolute value of each element of the original array.
        # if 28 in active_indices:
        #     print(1)
        return active_indices
    def calculate(self,n_cal=1,cal_feas = True, cal_opt = True):
        feas_results = [{'is_valid':False,'error':0.,'active_indices':np.array([], dtype=int) ,'x_org':np.zeros(self.dim)} for _ in range(n_cal)]
        opt_results = [{'is_valid':False,'error':0.,'active_indices':np.array([], dtype=int) ,'x_org':np.zeros(self.dim)} for _ in range(n_cal)]

        for i in range(n_cal):
            # Random optimization direction
            c = np.random.randn(self.dim)  #Generate random numbers or random arrays that satisfy the standard normal distribution
            # c = -np.ones(self.dim)
            if self.is_epigraph:
                # c[-1] = np.abs(c[-1])
                c[-1] = 1.

            # c = np.array([1,1])

            if cal_feas:
                # Feasible error path
                x_apx = self.optimize_direction(c, in_approx=True)
                x_org = self.project(x_apx) if x_apx is not None else None
                if x_apx is not None and x_org is not None:
                    e_feas = np.sum((x_apx - x_org) ** 2)
                    feas_results[i]['error'] = e_feas
                    if e_feas > self.params['feas_tol']:
                        feas_results[i]['is_valid'] = True
                        feas_results[i]['active_indices'] = self._find_active(x_apx)
                        feas_results[i]['x_org'] = x_org
                else:
                    print(self._iter)
            if cal_opt:
                # optimal error path
                x_org = self.optimize_direction(c)
                x_apx = self.project(x_org, to_approx=True) if x_org is not None else None
                if x_org is not None and x_apx is not None:
                    e_opt = np.sum((x_apx - x_org) ** 2)
                    opt_results[i]['error'] = e_opt
                    if e_opt > self.params['opt_tol']:
                        opt_results[i]['is_valid'] = True
                        opt_results[i]['active_indices'] = self._find_active(x_apx)
                        opt_results[i]['x_org'] = x_org
                        # print(self.A_hat[28,3])
                else:
                    print(self._iter)

        # Record error
        self.training_history['feas'].append(np.mean([feas_results[i]['error'] for i in range(n_cal)]))
        self.training_history['opt'].append(np.mean([opt_results[i]['error'] for i in range(n_cal)]))
        self._iter += 1
        return feas_results, opt_results
    def update_polytope(self,A_hat=None,b_hat=None):
        n_constraints = self.A_hat.shape[0]
        if A_hat is not None:
            self.A_hat = A_hat
            for i in range(n_constraints):
                for j in range(self.dim):
                    self.approx_model.A[i, j].value = self.A_hat[i, j]
        if b_hat is not None:
            self.b_hat = b_hat
            for i in range(n_constraints):
                self.approx_model.b[i].value = self.b_hat[i]
    def update_parameters(self, param_dict):
        """General parameter update function
        Args:
            model: Pyomomodel object
            param_dict: Parameter dictionary, supported formats:
                - Scalar: direct assignment
                - one dimensional: nparray or dictionary{Index: value}
                - two-dimensional: nparray or dictionary{(i,j): value}
        """
        for param_name, value in param_dict.items():
            param = getattr(self.original_model, param_name)  # Read the named model parameter.

            # Scalar parameter handling
            if not param.is_indexed():
                param.set_value(float(value))
                continue

            # Multidimensional parameter processing
            indices = param.index_set()

            # One-dimensional parameters (Index format is single element)
            if all(not isinstance(idx, tuple) for idx in indices):
                if isinstance(value, np.ndarray):
                    if value.ndim != 1:
                        raise ValueError(f"parameters {param_name} Requires one-dimensional array")
                    for i, idx in enumerate(indices):
                        param[idx].set_value(value[i])
                elif isinstance(value, dict):
                    for idx, val in value.items():
                        param[idx].set_value(val)
                else:
                    raise ValueError(f"Unsupported type: {type(value)}")

            # 2D parameters (Index format is tuple)
            else:
                if isinstance(value, np.ndarray):
                    if value.ndim != 2:
                        raise ValueError(f"parameters {param_name} Requires two-dimensional array")
                    for (i, j), idx in zip(np.ndindex(value.shape), indices):
                        param[idx].set_value(value[i, j])  # Follow NumPy index order.
                elif isinstance(value, dict):
                    for (i, j), val in value.items():
                        param[i, j].set_value(val)
                else:
                    raise ValueError(f"Unsupported type: {type(value)}")
        if self.is_epigraph:
            self._cal_max_obj()
            self.original_model.fmax.set_value(self.fmax)
    def copy(self):
        ec = ErrorCalculator(original_model=self.original_model_dict,
                            A_hat = self.A_hat,
                             b_hat= self.b_hat,
                             is_epigraph= self.is_epigraph,
                             solver=self.solver_str)
        ec.configure(**self.params)
        return ec

    # ===================== R3: reference-point radial search =====================
    # This optional extension promotes the radial construction used by the plotting
    # scripts to a training-time error probe. Along direction v, the optimizer finds
    # the true-domain radius k_omega, while the polytope radius k_poly is computed in
    # closed form. Both points use the same record format as calculate(), so the
    # feasibility/optimality loss remains unchanged. The feature is inactive unless
    # attach_radial() is called.
    def attach_radial(self, ppc, init_params_dict, n_dirs=8, seed=0,
                      pd_name='Pd_meta', qd_name='Qd_meta'):
        """Prepare the ray model and constants for radial-error evaluation.

        ``n_dirs`` controls the number of directions sampled at each step. Directions
        are regenerated during training so that the circle is covered progressively;
        a fixed set could leave unsampled angular gaps.
        _radial_dirs A copy is still generated for record purposes only./Fallback. Return to this example direction (n_dirs, 2)."""
        self._radial = True
        self._radial_ppc = ppc
        self._radial_init = init_params_dict
        self._radial_pd_name = pd_name
        self._radial_qd_name = qd_name
        self._radial_n_dirs = n_dirs               # Number of current mining directions per step (Trainer use)
        self._radial_dirs = self._gen_radial_dirs(n_dirs, seed)   # Fixed direction, only for record keeping/rollback
        if (ppc is not None and pd_name in init_params_dict
                and qd_name in init_params_dict):
            self._reference_point_calculator = ReferencePointCalculator(
                ppc=ppc,
                init_params_dict=init_params_dict,
                original_model=self.original_model,
                solver=self.solver_str,
            )
        else:
            # Analytic cases (TwoDiskEC/StarEC) override _compute_x_ref and do
            # not have distribution-system Pd_meta/Qd_meta parameters.
            self._reference_point_calculator = None
        self._radial_reference_outside_count = 0
        self._ray_model = self._build_ray_model()
        self._ray_solver = SolverFactory(self.solver_str, tee=False)
        return self._radial_dirs

    @staticmethod
    def _gen_radial_dirs(n_dirs, seed):
        """generate n_dirs two-dimensional unit direction vectors (transplanted from generate_random_directions) ; with seeds, fixed."""
        rng = np.random.RandomState(seed)
        ang = rng.uniform(0, 2 * np.pi, n_dirs)
        return np.column_stack((np.cos(ang), np.sin(ang)))

    @staticmethod
    def _gen_radial_dirs_fresh(n_dirs):
        """Every step of the way n_dirs unit direction (use global np.random status, not fixed seed).
        and calculate() of c=np.random.randn Homology - the direction gradually and densely covers the entire circle with training,
        No more fixed-direction gaps.R3 Use this for training (align the original"Every step of the way"design) ."""
        ang = np.random.uniform(0, 2 * np.pi, n_dirs)
        return np.column_stack((np.cos(ang), np.sin(ang)))

    def radial_dtheta(self, batch_data, i):
        """from batch_data Take the first place i sample Δθ, press [Pd_flat, Qd_flat] spelled in order dtheta vector.
        sequence with _compute_x_ref / _sync_ray_meta The position segmentation is strictly consistent (self-built and self-segmented, does not rely on dictionary order)) ."""
        pd = batch_data[self._radial_pd_name][i].detach().cpu().numpy().flatten()
        qd = batch_data[self._radial_qd_name][i].detach().cpu().numpy().flatten()
        return np.concatenate([pd, qd])

    def _build_ray_model(self):
        """clone original_model, Add ray constraints var_proj==x_ref+k·v with max-k target.
        Ported from approximate_polygon_coverage.create_ray_model."""
        ray = self.original_model.clone()
        dim = len(ray.var_proj)
        ray.ray_v = Param(range(dim), mutable=True, initialize=0.0)
        ray.ray_xref = Param(range(dim), mutable=True, initialize=0.0)
        ray.k = Var(bounds=(0, None), initialize=0.0)
        def _ray_con(m, j):
            return m.var_proj[j] == m.ray_xref[j] + m.k * m.ray_v[j]
        ray.ray_con = Constraint(range(dim), rule=_ray_con)
        ray.min_direction.deactivate()
        ray.min_error.deactivate()
        ray.max_k = Objective(expr=ray.k, sense=maximize)
        return ray

    def _compute_x_ref(self, dtheta):
        """Find the no-flexibility reference point under the same operating condition ``[P0, Q0]`` (per unit value) ."""
        if self._reference_point_calculator is None:
            raise RuntimeError("The current study example is not configured with a distribution network reference point solver.")
        return self._reference_point_calculator.compute(dtheta)

    def _sync_ray_meta(self, dtheta):
        """put the current dtheta Corresponding Pd_meta/Qd_meta Push to ray model. Ported from update_ray_model_params."""
        init = self._radial_init
        Pd_init = init['Pd_meta']['initial_value']
        Qd_init = init['Qd_meta']['initial_value']
        dtheta = np.asarray(dtheta)
        Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
        Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)
        ray = self._ray_model
        bus_ids = list(ray.BUS)
        if Pd_init.ndim == 2:
            phases = ['a', 'b', 'c']
            for i, bus in enumerate(bus_ids):
                for p, ph in enumerate(phases):
                    ray.Pd_meta[bus, ph] = float(Pd_meta_new[i, p])
                    ray.Qd_meta[bus, ph] = float(Qd_meta_new[i, p])
        else:
            for i, bus in enumerate(bus_ids):
                ray.Pd_meta[bus] = float(Pd_meta_new.flat[i])
                ray.Qd_meta[bus] = float(Qd_meta_new.flat[i])

    def _solve_k_omega(self, x_ref, v):
        """ipopt Solution original feasible region along v The radial maximum k (constant target, Yes (A,b) no gradient).
        Ported from compute_k_max_ray."""
        ray = self._ray_model
        dim = len(ray.var_proj)
        for j in range(dim):
            ray.ray_xref[j] = float(x_ref[j])
            ray.ray_v[j] = float(v[j])
        ray.k.value = 0.0
        try:
            res = self._ray_solver.solve(ray)
            if res.solver.termination_condition == TerminationCondition.optimal:
                return ray.k.value
        except Exception:
            pass
        return 0.0

    def calculate_radial(self, v_dirs, dtheta):
        """Calculates a reference point for a given set of directions x_ref with all directions k_Ω (constant target) .
        Return list[dict] (one in each direction), including 'v'/'x_ref'/'k_omega'/'is_valid_radial';
        Reference point solution fails and returns None (Trainer / loss end skips the sample) ."""
        x_ref = self._compute_x_ref(dtheta)
        if x_ref is None:
            return None
        self._sync_ray_meta(dtheta)
        out = []
        for v in v_dirs:
            k = self._solve_k_omega(x_ref, v)
            out.append({'v': np.asarray(v, dtype=float),
                        'x_ref': np.asarray(x_ref, dtype=float),
                        'k_omega': float(k),
                        'is_valid_radial': bool(k > 1e-9)})
        return out

    def _polytope_radial_k(self, x_ref, v):
        """Polyhedron self.A_hat x<=self.b_hat from x_ref along unit direction v The radial maximum k (numpy, use freeze A_hat/b_hat) .
        Return (k_P, i_active).closed type k = min_i (b_i−A_i·x̂)/(A_i·v) (A_i·v≤tol set inf) : Use freeze snapshot here active
        constraint, calculation k_P used for feas/opt classification; the true gradient is given by compute_loss In Kewei (A_pred,b_pred) Good deal."""
        A, b = self.A_hat, self.b_hat
        Av = A @ v
        Ax = A @ x_ref
        ratio = (b - Ax) / (Av + 1e-12)
        ratio = np.where(Av > 1e-9, ratio, np.inf)   # A_i·v≤tol The constraints are not bounded +v direction, location inf
        i_active = int(np.argmin(ratio))
        return float(ratio[i_active]), i_active

    def calculate_radial_feasopt(self, dtheta, v_dirs):
        """R3 Reference point radial [probe replacement] version: return and calculate() same format (feas_results, opt_results),
        Feed directly [original compute_loss]——feas/opt mechanism, rate_opt_feas The weights and two-stage structure are identical to the original manuscript.
        Just put"detection Ω way"Change from the support point (convex hull, only convex can be seen) to the reference point radial (k_Ω, Non-convex depressions can be seen).

        per direction v:
          x̂+k_Ω·v = Ω radial boundary point (target, ipopt Ask for k_Ω, Constant, no gradient)
          x̂+k_P·v = Polygon Radial Boundary Points (active Constraints here, k_P frozen by A_hat/b_hat closed type)
          k_P>k_Ω (Polygon exceeds Ω = Not feasible) → feas Sample: Pull activation constraints inwards to ensure safety;
          k_P<k_Ω (Polygon not arrived Ω = suboptimal)  → opt  Sample: Push out the activation constraints to obtain completeness.
        compute_loss in active constraint calculation (A·target−b)², Pull the activation constraint to x̂+k_Ω·v (ρ=k_P/k_Ω→1) .
        """
        dim = self.dim

        def _blank():
            return {'is_valid': False, 'error': 0.,
                    'active_indices': np.array([], dtype=int),
                    'x_org': np.zeros(dim)}

        x_ref = self._compute_x_ref(dtheta)
        if x_ref is None:
            return [_blank() for _ in v_dirs], [_blank() for _ in v_dirs]
        is_member, _ = reference_point_membership(
            self.A_hat, self.b_hat, x_ref, REFERENCE_MEMBERSHIP_TOL
        )
        if not is_member:
            # k_P is not a radial length when the ray starts outside P.
            self._radial_reference_outside_count += 1
            blanks_feas = [_blank() for _ in v_dirs]
            blanks_opt = [_blank() for _ in v_dirs]
            self.training_history['feas'].append(0.0)
            self.training_history['opt'].append(0.0)
            self._iter += 1
            return blanks_feas, blanks_opt
        self._sync_ray_meta(dtheta)

        feas_results, opt_results = [], []
        for v in v_dirs:
            k_omega = self._solve_k_omega(x_ref, v)          # Ω Radial arrival (ipopt, constant target)
            if k_omega <= 1e-9:
                feas_results.append(_blank()); opt_results.append(_blank()); continue
            k_P, _ = self._polytope_radial_k(x_ref, v)       # Polygon radial reach (frozen snapshot closed form)
            if not np.isfinite(k_P):
                feas_results.append(_blank()); opt_results.append(_blank()); continue
            x_apx = x_ref + k_P * v                          # Polygon Radial Boundary Points (active at)
            x_org = x_ref + k_omega * v                      # Ω radial boundary point (target)
            active_indices = self._find_active(x_apx)        # Yohara calculate() Same law
            rec_feas, rec_opt = _blank(), _blank()
            if k_P > k_omega + 1e-9:                         # beyond Ω = Not feasible → feas
                rec_feas.update(is_valid=True, error=(k_P - k_omega) ** 2,
                                active_indices=active_indices, x_org=x_org)
            elif k_P < k_omega - 1e-9:                       # Didn’t arrive Ω = suboptimal → opt
                rec_opt.update(is_valid=True, error=(k_omega - k_P) ** 2,
                               active_indices=active_indices, x_org=x_org)
            # else Fitted (ρ≈1) → both invalid, No gradient
            feas_results.append(rec_feas)
            opt_results.append(rec_opt)
        # with calculate() Consistent: record feas/opt average error + advance _iter, supply callback Print
        #  (If you don’t write it training_history is empty, callback of np.mean([]) → FeasErr/OptErr show nan)
        self.training_history['feas'].append(np.mean([r['error'] for r in feas_results]))
        self.training_history['opt'].append(np.mean([r['error'] for r in opt_results]))
        self._iter += 1
        return feas_results, opt_results


class PreTrainNet(nn.Module):  #A, bThe training has nothing to do with adjustable parameters
    def __init__(self,A_init,b_init,is_epigraph = False,device = 'cpu'):
        super().__init__()
        self.nrows,self.dim_x = A_init.shape  # (8, 2)
        self.is_epigraph = is_epigraph
        if is_epigraph:
            self.A_net = nn.Sequential(
                ZeroInitLinear(0, (self.nrows-1)*self.dim_x, init_bias=np.array(A_init[:-1]).flatten(),device = device)
            )
        else:
            self.A_net = nn.Sequential(         #8*2=16                            (16)
                ZeroInitLinear(0, self.nrows*self.dim_x, init_bias=np.array(A_init).flatten(),device = device)  #.flatten()Array flattening => One-dimensional tensor
            )
        self.b_net = nn.Sequential(
            ZeroInitLinear(0, self.nrows, init_bias=np.array(b_init), device=device)
        )

    def forward(self):
        # Generate the constraint matrix.
        A_flat = self.A_net(torch.empty(0))
        if self.is_epigraph:
            A = A_flat.view(-1, self.nrows-1, self.dim_x)  # shape: (batch_size, 2, 2)
            final_row = torch.tensor(np.hstack([np.zeros(self.dim_x-1), [1.0]]),dtype=torch.float32, device=A.device).reshape(-1,1,self.dim_x)
            A = torch.cat([A, final_row], dim=1)
        else:
            A = A_flat.view(-1, self.nrows, self.dim_x)  # One matrix per batch item.
        # Normalize each row of A.
        row_norms = torch.norm(A, dim=2, keepdim=True)  # shape: (batch_size, nrows, 1)
        A_normalized = A / (row_norms + 1e-8)  # Add small epsilon to avoid division by zero

        # Generate the constraint vector.
        b = self.b_net(torch.empty(0))  # shape: (batch_size, 2)

        # Normalize each entry of b by the corresponding row norm of A.
        b_normalized = b / (row_norms.squeeze(2) + 1e-8)  # shape: (batch_size, 2)  #.squeeze(2) That is: delete those redundant ones with a length of1Dimensions

        return A_normalized, b_normalized

class BiasNet(nn.Module):
    def __init__(self,dim_theta,b_init,n_hidden = 64,hidden_sizes=None,activation='relu',device = 'cpu'):
        super().__init__()
        self.nrows = b_init.shape[0]
        if dim_theta:
            if hidden_sizes is None:
                hidden_sizes = [n_hidden]   # Not passed on → single layer(original condition), Completely identical to the original implementation
            # ---- Original implementation (single hidden layer + ReLU) , Keep for future reference ----
            # self.b_net = nn.Sequential(
            #     nn.Linear(dim_theta, n_hidden),
            #     nn.ReLU(),
            #     ZeroInitLinear(n_hidden, self.nrows, init_bias=np.array(b_init),device = device)
            # )
            # ---- New implementation: support for configurable number of hidden layers + activation function ----
            self.b_net = build_mlp(dim_theta, hidden_sizes, activation,
                                   self.nrows, np.array(b_init), device=device)
        else:
            self.b_net = nn.Sequential(
                ZeroInitLinear(0, self.nrows, init_bias=np.array(b_init),device = device)
            )

    def forward(self, delta_theta):
        b = self.b_net(delta_theta)
        return b

class FullNet(nn.Module):
    def __init__(self,dim_theta,A_init,b_init,is_epigraph = False,n_hidden = 64,hidden_sizes=None,activation='relu',device = 'cpu'):
        super().__init__()
        self.nrows,self.dim_x = A_init.shape
        self.is_epigraph = is_epigraph
        if hidden_sizes is None:
            hidden_sizes = [n_hidden]   # Not passed on → single layer(original condition), Completely identical to the original implementation
        # ---- Original implementation (single hidden layer + ReLU) , Keep for future reference ----
        # if is_epigraph:
        #     self.A_net = nn.Sequential(
        #         nn.Linear(dim_theta, n_hidden),
        #         nn.ReLU(),
        #         ZeroInitLinear(n_hidden, (self.nrows-1)*self.dim_x, init_bias=np.array(A_init[:-1]).flatten(),device = device)
        #     )
        # else:
        #     self.A_net = nn.Sequential(
        #         nn.Linear(dim_theta, n_hidden),
        #         nn.ReLU(),
        #         ZeroInitLinear(n_hidden, self.nrows*self.dim_x, init_bias=np.array(A_init).flatten(),device = device)
        #     )
        # self.b_net = nn.Sequential(
        #     nn.Linear(dim_theta, n_hidden),
        #     nn.ReLU(),
        #     ZeroInitLinear(n_hidden, self.nrows, init_bias=np.array(b_init), device=device)
        # )
        # ---- New implementation: support for configurable number of hidden layers + activation function ----
        if is_epigraph:
            self.A_net = build_mlp(dim_theta, hidden_sizes, activation,
                                   (self.nrows-1)*self.dim_x,
                                   np.array(A_init[:-1]).flatten(), device=device)
        else:
            self.A_net = build_mlp(dim_theta, hidden_sizes, activation,
                                   self.nrows*self.dim_x,
                                   np.array(A_init).flatten(), device=device)
        self.b_net = build_mlp(dim_theta, hidden_sizes, activation,
                               self.nrows, np.array(b_init), device=device)


    def forward(self, delta_theta):
        # Generate the constraint matrix.
        A_flat = self.A_net(delta_theta)

        if self.is_epigraph:
            A = A_flat.view(-1, self.nrows-1, self.dim_x)  # shape: (batch_size, 2, 2)
            batch_size = A.shape[0]
            final_row = torch.tensor(np.hstack([np.zeros(self.dim_x-1), [1.0]]),dtype=torch.float32, device=A.device).reshape(-1,1,self.dim_x)
            final_row_batch = final_row.repeat(batch_size,1,1)
            A = torch.cat([A, final_row_batch], dim=1)
        else:
            A = A_flat.view(-1, self.nrows, self.dim_x)  # shape: (batch_size, 2, 2)
        # Normalize each row of A.
        row_norms = torch.norm(A, dim=2, keepdim=True)  # shape: (batch_size, 2, 1)
        A_normalized = A / (row_norms + 1e-8)  # Add small epsilon to avoid division by zero

        # Generate the constraint vector.
        b = self.b_net(delta_theta)  # shape: (batch_size, 2)

        # Normalize each entry of b by the corresponding row norm of A.
        b_normalized = b / (row_norms.squeeze(2) + 1e-8)  # shape: (batch_size, 2)

        return A_normalized, b_normalized

class ZeroInitLinear(nn.Module):
    """Fully connected layer with zero weights and a prescribed initial bias."""

    def __init__(self, in_features, out_features, init_bias,device='cpu'):
        super().__init__()  #Call the constructor of the parent class to complete the initialization of the parent class.
        self.device = device                      #16            0    128
        self.weight = nn.Parameter(torch.zeros(out_features, in_features,device=device))  #nn.Parameterwill not be trainable Tensor Convert to trainable parameters and register them as part of the model.
        self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32,device=device))  #torch.tensor(arrayarray) =>Tensor
                                                # (16)
    def forward(self, x):
        return torch.matmul(x.to(self.device), self.weight.t()) + self.bias


# activation function table + MLP construction factory (for FullNet/BiasNet Configurable hidden layer of)
ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'silu': nn.SiLU,
}

def build_mlp(in_features, hidden_sizes, activation, out_features, init_bias, device='cpu'):
    """build Linear→Act→...→ZeroInitLinear The multilayer perceptron body.
    hidden_sizes: Hidden layer width list, [128]=single layer(original condition), [128,64]=Double layer
    activation: Unified activation name string(see ACTIVATIONS); or a list of activation names for each layer(heterogeneous combination),
        Such as ['relu','tanh'] Represents the first layer ReLU, second floor Tanh (The length must be equal to hidden_sizes consistent).
        pass str when all hidden layers share the same activation(Original behavior, backwards compatible main_ds.py).
    The final layer is fixed to ZeroInitLinear(in=Hidden width of last layer, out=out_features, bias=init_bias)
    """
    layers, prev = [], in_features
    # Original implementation (unified activation) : Act = ACTIVATIONS[activation]; Common to all hidden layers Act()
    # Now supported str (unified) or list (Different types of combinations, one for each layer) , str Path behavior unchanged
    if isinstance(activation, str):
        acts = [ACTIVATIONS[activation]] * len(hidden_sizes)
    else:
        assert len(activation) == len(hidden_sizes), \
            f"activation List length {len(activation)} with hidden_sizes length {len(hidden_sizes)} inconsistent"
        acts = [ACTIVATIONS[a] for a in activation]
    for h, Act in zip(hidden_sizes, acts):
        layers += [nn.Linear(prev, h), Act()]
        prev = h
    layers.append(ZeroInitLinear(prev, out_features, init_bias=init_bias, device=device))
    return nn.Sequential(*layers)


# Vector method to calculate loss function
def compute_loss(A, b, batch_data):
    # Extract data and convert to tensor
    batch_size = len(batch_data)
    if batch_size == 0:
        return torch.tensor(0.0, device=A.device)

    n_cal = len(batch_data[0])
    m, n = A.shape[-2], A.shape[-1]
    device = A.device

    # Initialize storage tensor
    x_org_list = []
    is_valid_list = []
    active_indices_coords = []

    # Traversebatch_dataExtract information
    for batch_idx, batch in enumerate(batch_data):
        x_batch = []
        is_valid_batch = []
        for ncal_idx, data in enumerate(batch):
            # x_orgProcess
            x_org = torch.from_numpy(data['x_org']).float().to(device)
            x_batch.append(x_org)

            # is_validProcess
            is_valid_batch.append(data['is_valid'])

            # active_indicesCoordinate collection
            active_indices = torch.from_numpy(data['active_indices']).long().to(device)
            for idx in active_indices:
                active_indices_coords.append((batch_idx, ncal_idx, idx.item()))

        x_org_list.append(torch.stack(x_batch))
        is_valid_list.append(torch.tensor(is_valid_batch, device=device))

    # Construct core tensor
    x_org = torch.stack(x_org_list)  # (batch_size, n_cal, n)
    is_valid = torch.stack(is_valid_list).bool()  # (batch_size, n_cal)

    # generateactive_indicesof0-1mask matrix
    mask = torch.zeros(batch_size, n_cal, m, device=device)
    if active_indices_coords:
        batch_idx, ncal_idx, active_idx = zip(*active_indices_coords)
        batch_idx = torch.tensor(batch_idx, device=device)
        ncal_idx = torch.tensor(ncal_idx, device=device)
        active_idx = torch.tensor(active_idx, device=device)
        mask[batch_idx, ncal_idx, active_idx] = 1.0

    # Calculate predicted value (Vectorized implementation)
    A_expanded = A.unsqueeze(1)  # (batch, 1, m, n)
    x_org_expanded = x_org.unsqueeze(3)  # (batch, n_cal, n, 1)
    mask_expanded = mask.unsqueeze(3)  # (batch, n_cal, m, 1)

    # Matrix multiplication (batch, n_cal, m, n) × (batch, n_cal, n, 1) → (batch, n_cal, m, 1)
    pred = torch.matmul(A_expanded * mask_expanded, x_org_expanded).squeeze(-1)  # *Element-wise multiplication

    # Calculate residuals
    b_expanded = b.unsqueeze(1).expand(-1, n_cal, -1)  # (batch, n_cal, m)
    b_masked = b_expanded * mask
    residual = pred - b_masked

    # Calculate squared loss and apply validity mask
    squared_loss = residual.pow(2) * mask  # (batch, n_cal, m)
    loss_per_sample = squared_loss.sum(dim=-1)  # (batch, n_cal)
    valid_loss = loss_per_sample * is_valid.float()

    # Calculate average loss
    total_loss = valid_loss.sum()
    num_valid = is_valid.sum().float()
    # print((num_valid,total_loss))
    # return total_loss / num_valid.clamp(min=1e-6)
    return total_loss / num_valid.clamp(min=1)


# Loop calculation loss function
# def compute_loss(A, b, batch_data):
#     """
#     Calculate the loss function
#
#     parameters:
#         A: tensor, shape (batch_size, m, n)
#         b: tensor, shape (batch_size, m)
#         batch_data: list of lists of dictionaries
#
#     Return:
#         loss: scalar tensor
#     """
#     batch_size = len(batch_data)
#     total_loss = 0.0
#     valid_count = 0
#     device = A.device
#     for i in range(batch_size):  # Iterate through eachbatch
#         for j in range(len(batch_data[i])):  # Iterate through eachn_calelement
#             data = batch_data[i][j]
#             # if not data['is_valid']:
#             #     continue  # Skip invalid data
#
#             active_indices = data['active_indices']
#             x_org = torch.tensor(data['x_org'], dtype=torch.float32, device=device)
#
#             # CalculateA_sub * x_org
#             A_subx = torch.matmul(A[i, active_indices, :], x_org)  # shape (len(active_indices),)
#
#             # Get the correspondingbvalue
#             b_sub = b[i, active_indices]  # shape (len(active_indices),)
#
#             # Calculate squared error
#             error = torch.sum((A_subx - b_sub)  ** 2)
#             total_loss += error
#             valid_count += 1
#
#     if valid_count == 0:
#         return torch.tensor(0.0, device=device,requires_grad=True)  # avoid dividing by zero
#
#     # Calculate average loss
#     loss = total_loss / valid_count
#     return loss

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            error_calculator: ErrorCalculator,
            compute_loss: Callable,  #Callable: callable
            params: Optional[Dict] = None,
            device = 'cpu',
    ):
        """
        Minimalist version of trainer
        - configure() Update configuration
        - initialize() Initialize component
        - train() Execute training
        """
        self.model = model
        self.error_calculator = error_calculator
        self.compute_loss = compute_loss
        self.loss_history = []
        self.grad_history = []
        # Keep these counters across initialize() calls to audit the full training budget.
        self.probe_statistics = {
            'optimizer_steps': 0,
            'training_samples': 0,
            'effective_training_samples': 0,
            'direction_probes': 0,
            'true_domain_oracle_calls': 0,
            'polytope_oracle_calls': 0,
            'feas_valid_samples': 0,
            'opt_valid_samples': 0,
            'valid_supervision_records': 0,
        }
        # Default parameter configuration
        self.default_params = {
            "optimizer": "SGD",
            "lr": 5e-1,
            "scheduler": None,
            "batch_size": 1,
            "n_cal": 1,
            "cal_feas": True,
            "cal_opt": True,
            "feas_tol": 1e-8,
            "opt_tol": 1e-8,
            "call_interval": 20,
            "training_callback": None,
            "radial_probe": False  # Use reference-point radial probes instead of support-point probes.
        }

        # Merge user parameters
        self.params = {**self.default_params,  ** (params or {})}

    def configure(self, **kwargs):
        """Dynamically update configuration parameters"""
        self.params.update(kwargs)

    def initialize(self):
        """Initialize all components with one click"""
        # Configuration error calculator
        self.error_calculator.configure(
            feas_tol=self.params["feas_tol"],
            opt_tol=self.params["opt_tol"],
            cal_feas = self.params["cal_feas"],
            cal_opt = self.params["cal_opt"]
        )

        # Optimizer settings
        opt_type = self.params["optimizer"].lower()
        base_lr = self.params["lr"]  # Default learning rate
        lr_A = self.params.get("lr_A", base_lr)  # Prefer dedicated learning rates
        lr_b = self.params.get("lr_b", base_lr)
        # Automatically build parameter groups
        param_groups = []
        if hasattr(self.model, 'A_net'):
            param_groups.append({'params': self.model.A_net.parameters(), 'lr': lr_A})
        if hasattr(self.model, 'b_net'):
            param_groups.append({'params': self.model.b_net.parameters(), 'lr': lr_b})
        # If no subnet is found, an error is reported.
        if not param_groups:
            raise RuntimeError("Not found in modelA_netorb_net")
        # Create optimizer
        if opt_type == "sgd":
            self.optimizer = optim.SGD(param_groups)
            # self.optimizer = optim.SGD(param_groups, momentum=0.95, nesterov=True)
        elif opt_type == "adam":
            self.optimizer = optim.Adam(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

        # Configure the optional learning-rate scheduler.
        self.scheduler = None
        if self.params["scheduler"]:
            config = self.params["scheduler"]
            if config["type"] == "StepLR":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config.get("step_size", 100),
                    gamma=config.get("gamma", 0.95)
                )

    def train(self, n_train: int = 1, params_data = None, parallel:bool = False):
        if not parallel:
            self._train_serial(n_train=n_train, params_data = params_data)
        else:
            self._train_parallel(n_train=n_train, params_data = params_data)
    def _train_serial(self, n_train: int = None, params_data = None):
        """Universal training entrance"""
        if not isinstance(self.model, (PreTrainNet,BiasNet,FullNet)):
            warnings.warn(f"Not supported yet {type(self.model).__name__} Model training, skipped")
            return
            """Complete training cycle"""
        i_iter = 0
        if isinstance(self.model, PreTrainNet):
            params_data['dataloader'] = [None]
        for epoch in range(n_train+1):
        #for epoch in range(n_train):
            for batch_data in params_data['dataloader']:  # batch_dataThe structure depends ondatasetReturned content,Usually (data, label) or dictionary /tuple form
                # start_time = time.time()  # Start timing
                # Gradient clear
                self.optimizer.zero_grad()

                # forward propagation
                if isinstance(self.model, PreTrainNet):
                    A_pred, b_pred = self.model()
                    A_pred = A_pred.repeat(self.params["batch_size"], 1, 1)   #torch.tensor.repeat()Functions can repeatedly expand tensors
                    b_pred = b_pred.repeat(self.params["batch_size"], 1)
                elif isinstance(self.model, BiasNet):
                    b_pred = self.model(self._combine_batch(batch_data))
                    A_pred = self.params["A_pretrained"].repeat(self.params["batch_size"], 1, 1)
                else:
                    A_pred,b_pred = self.model(self._combine_batch(batch_data))  #After training, we getAandb

                # Batch processing
                feas_results, opt_results = [], []
                # R3 optionally replaces support-point probes with reference-point
                # radial probes while retaining the same loss calculation.
                radial_probe = (self.params.get("radial_probe", False)
                                and getattr(self.error_calculator, "_radial", False))
                for i in range(self.params["batch_size"]):
                    # Update polyhedral parameters
                    self.error_calculator.update_polytope(
                        A_hat=A_pred[i].detach().cpu().numpy(),
                        b_hat=b_pred[i].detach().cpu().numpy()
                    )
                    # Update model parameters
                    if batch_data is not None:
                        params_upd = {}
                        for name, data in batch_data.items():
                            params_upd[name] = params_data['params_dict'][name]['initial_value']+data[i].detach().cpu().numpy()  #add noise

                        self.error_calculator.update_parameters(params_upd)

                    # Calculation result: projected version calculate() (Support point detection, only convex hull can be seen) ;
                    # Radial probes expose non-convexity while preserving the loss-record format.
                    if radial_probe and batch_data is not None:
                        dtheta_i = self.error_calculator.radial_dtheta(batch_data, i)
                        n_dir = getattr(self.error_calculator, '_radial_n_dirs', 8)
                        dirs_i = self.error_calculator._gen_radial_dirs_fresh(n_dir)
                        f, o = self.error_calculator.calculate_radial_feasopt(dtheta_i, dirs_i)
                    else:
                        f, o = self.error_calculator.calculate(
                            n_cal=self.params["n_cal"],
                            cal_feas=self.params["cal_feas"],
                            cal_opt=self.params["cal_opt"]
                        )
                    n_probe = len(f)
                    n_feas_valid = sum(bool(rec['is_valid']) for rec in f)
                    n_opt_valid = sum(bool(rec['is_valid']) for rec in o)
                    self.probe_statistics['training_samples'] += 1
                    self.probe_statistics['effective_training_samples'] += int(
                        n_feas_valid + n_opt_valid > 0)
                    self.probe_statistics['direction_probes'] += n_probe
                    self.probe_statistics['feas_valid_samples'] += n_feas_valid
                    self.probe_statistics['opt_valid_samples'] += n_opt_valid
                    self.probe_statistics['valid_supervision_records'] += (
                        n_feas_valid + n_opt_valid)
                    if radial_probe:
                        # Query each radial boundary once per direction.
                        self.probe_statistics['true_domain_oracle_calls'] += n_probe
                        self.probe_statistics['polytope_oracle_calls'] += n_probe
                    else:
                        # The projection method issues one query per enabled loss term and direction.
                        n_objectives = (int(bool(self.params['cal_feas']))
                                        + int(bool(self.params['cal_opt'])))
                        self.probe_statistics['true_domain_oracle_calls'] += (
                            n_probe * n_objectives)
                        self.probe_statistics['polytope_oracle_calls'] += (
                            n_probe * n_objectives)
                    feas_results.append(f)
                    opt_results.append(o)

                # # Calculate double loss
                loss_feas = self.compute_loss(A_pred, b_pred, feas_results)
                loss_opt = self.params['rate_opt_feas'] * self.compute_loss(A_pred, b_pred, opt_results)
                #
                # # Backpropagation
                # loss_feas.backward(retain_graph=True)
                # loss_opt.backward()

                # Calculate double loss (with original main_ds.py completely consistent: loss_feas + rate_opt_feas·loss_opt, No extras;
                # Radial probing changes only the feasibility/optimality records, not the loss formula.
                loss_total = loss_feas + loss_opt

                # Backpropagation
                loss_total.backward()

                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2e6)
                self.grad_history.append(np.max(np.abs(self.model.A_net[0].bias.grad.detach().cpu().numpy())))
                # Parameter update
                self.optimizer.step()
                self.probe_statistics['optimizer_steps'] += 1
                if self.scheduler:
                    self.scheduler.step()

                # record loss
                self.loss_history.append((loss_feas.item(), loss_opt.item()))

                # trigger callback
                if i_iter % self.params["call_interval"] == 0:
                    # print(np.max(self.grad_history[-min(5,len(self.grad_history)):]))
                    # print(self.loss_history[-1])
                    callback = self.params["training_callback"]
                    if callback:
                        callback(self.error_calculator,i_iter)
                i_iter += 1
                # end_time = time.time()
                # elapsed = end_time - start_time  # Calculation time (seconds)
                # print(f"Iterate{i_iter}, Time consuming: {elapsed:.4f} seconds")
    def _train_parallel(self, n_train: int = None, params_data = None):
        """Universal training entrance"""
        if not isinstance(self.model, (PreTrainNet, BiasNet, FullNet)):
            warnings.warn(f"Not supported yet {type(self.model).__name__} Model training, skipped")
            return
        # ================ Initialize parallelization facilities ================
        batch_size = self.params["batch_size"]
        error_calculators = []
        for _ in range(batch_size):
            ec = self.error_calculator.copy()
            error_calculators.append(ec)
            # 1. Pre-create multiple independenterror_calculatorExample
        def process_task(args):
            i, A_hat, b_hat, params_upd = args
            # Directly use the pre-generatediinstances
            ec = error_calculators[i]
            ec.update_polytope(A_hat, b_hat)
            if params_upd is not None:
                ec.update_parameters(params_upd)
            f, o = ec.calculate(
                n_cal=self.params["n_cal"],
                cal_feas=self.params["cal_feas"],
                cal_opt=self.params["cal_opt"]
            )
            return (f, o)
        i_iter = 0
        if isinstance(self.model, PreTrainNet):
            params_data['dataloader'] = [None]
        for epoch in range(n_train):
            for batch_data in params_data['dataloader']:
                # start_time = time.time()  # Start timing
                self.optimizer.zero_grad()

                if isinstance(self.model, PreTrainNet):
                    A_pred, b_pred = self.model()
                    A_pred = A_pred.repeat(self.params["batch_size"], 1, 1)
                    b_pred = b_pred.repeat(self.params["batch_size"], 1)
                elif isinstance(self.model, BiasNet):
                    b_pred = self.model(self._combine_batch(batch_data))
                    A_pred = self.params["A_pretrained"].repeat(self.params["batch_size"], 1, 1)
                else:
                    A_pred, b_pred = self.model(self._combine_batch(batch_data))

                with ThreadPoolExecutor(max_workers=5) as executor:
                    # Bind index when generating task parametersi
                    params_dict_list = [None for _ in range(self.params["batch_size"])]
                    for i in range(self.params["batch_size"]):
                        params_upd = {}
                        if batch_data is not None:
                            for name, data in batch_data.items():
                                params_upd[name] = params_data['params_dict'][name]['initial_value'] + data[i].detach().cpu().numpy()
                        params_dict_list[i] = params_upd
                    task_args = [
                        (i, A_pred[i].detach().cpu().numpy(),
                         b_pred[i].detach().cpu().numpy(),
                         params_dict_list[i])
                        for i in range(self.params["batch_size"])
                    ]

                    # Submit tasks and get results
                    results = executor.map(process_task, task_args)
                    feas_results, opt_results = zip(*results)

                # Subsequent loss calculations and backpropagation remain unchanged
                loss_feas = self.compute_loss(A_pred, b_pred, feas_results)
                loss_opt = self.params['rate_opt_feas'] * self.compute_loss(A_pred, b_pred, opt_results)
                #
                # loss_feas.backward(retain_graph=True)
                # loss_opt.backward()

                # Calculate double loss
                loss_total = loss_feas + loss_opt

                # Backpropagation
                loss_total.backward()

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                self.loss_history.append((loss_feas.item(), loss_opt.item()))

                if i_iter % self.params["call_interval"] == 0:
                    if self.params["training_callback"]:
                        self.params["training_callback"](error_calculators[0], i_iter)
                i_iter += 1
                # end_time = time.time()
                # elapsed = end_time - start_time  # Calculation time (seconds)
                # print(f"Iterate{i_iter}, Time consuming: {elapsed:.4f} seconds")

    def _combine_batch(self,batch_dict):
        if not batch_dict:
            return torch.empty((1,0))
        features = []
        for key in batch_dict.keys():
            tensor = batch_dict[key]
            # Flatten all dimensions except the batch dimension (e.g. (5,2,3) flattened to (5,6))
            flattened = tensor.view(tensor.size(0), -1)  # Preserve the batch dimension.
            features.append(flattened)
        return torch.cat(features, dim=1)

def pyomo_params_to_numpy(model):
    params_dict = {}
    param_count = 0

    def _get_matrix_dimensions(matrix_param):
        rows = set()
        cols = set()
        for (i, j) in matrix_param.index_set():
            rows.add(i)
            cols.add(j)
        return len(rows), len(cols)

    for param in model.component_objects(Param, descend_into=True):
        if not param.mutable:
            continue
        param_name = param.name
        if not param.is_indexed():  # 0dimension parameter (scalar)
            value = param.value
            size = 1
            param_count+=size
        else:  # Multidimensional parameters
            # Get parameter dimensions
            dim = param.dim()
            if dim == 1:  # 1Dimension parameters (vector)
                values = [param[idx].value for idx in param.index_set()]
                value = np.array(values)
                size = len(value)
                param_count+=size
            elif dim == 2:  # 2Dimension parameters (matrix)
                nrow,ncol = _get_matrix_dimensions(param)

                # Create a matrix and fill it with values
                value = np.zeros((nrow, ncol))
                for count, idx in enumerate(param.index_set()):
                    i,j = count//ncol,count%ncol
                    value[i, j] = param[idx[0], idx[1]].value
                size = (nrow,ncol)
                param_count+=size[0]*size[1]
            else:
                raise ValueError(f"parameters {param.name} Dimensions {dim} exceed2Dimension, this code does not support")
        # Store parameter information in the dictionary
        params_dict[param_name] = {
            'initial_value': value,
            'size': size
        }
    return params_dict, param_count
