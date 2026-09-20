# -*- coding: utf-8 -*-
"""Run the R1.1 two-module learning comparison on independent operating conditions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from pyomo.core.expr.calculus.derivatives import differentiate, Modes
import torch
from torch import nn
import torch.nn.functional as F

from Simulator import PROJECT_ROOT
from Simulator.runners import main_revise1_6_supervised as shared
from Simulator.runners.common_case33_test_conditions import (
    CONDITION_PATH as COMMON_TEST_CONDITION_PATH,
    load_common_test_conditions,
)


CASENAME = "case33bw_ds"
N_SIDES = 36                  # withPINNMaintain the same two-dimensional polygon expression capabilities.
INITIAL_POINTS = 10_000       # Papercase33Settings; Uniform candidates are balanced by label for each5000point.
UPDATE_POINTS_PER_ROUTE = 10
UPDATE_ROUNDS = 50
BETA_FRACTION = 0.01
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 3e-4
MAX_LEARNING_STEPS = 2500
LOSS_CHECK_INTERVAL = 20
FEASIBILITY_SLACK_TOL = 1e-7
POLYTOPE_CLASSIFICATION_TOL = 1e-7
SAMPLING_EXPANSION_FACTOR = 1.4
SAMPLING_CLASS_SCARCITY_THRESHOLD = 0.30
MIN_AXIS_RADIUS = 1e-3
ROW_NORM_TOL = 1e-10
SCORE_SCALE_MIN = 1e-3
SCORE_SCALE_MAX = 1e4
# Each normal may rotate only within its initial 10-degree angular sector.
# Keep adjacent sectors 1 degree apart within +/-4.5 degrees to avoid duplicate normals and unbounded polygons.
NORMAL_SECTOR_FRACTION = 0.45
KKT_COMPLEMENTARITY_TOL = 1e-6
# IPOPTWhen the direct search for exact complement fails, first find the relaxationMPCC, back againtau=0of originalKKT.
# Relaxed solutions serve only as initial values; formally accepted solutions must still pass exact complementary residual checks.
KKT_RELAXATION_SCHEDULE = (1e-3, 1e-4, 1e-5, 1e-6, 0.0)
KKT_POLYTOPE_CONTINUATION = (0.25, 0.50, 0.75, 1.0)
MAX_SAMPLE_ATTEMPT_FACTOR = 200
# later stageIF2FThe area is very thin: it is already fully mined in the official shards10Needed49attempts, 50times
# The upper limit appears again8/10critical failure. Raise to100Only expand multiple initial value searches at the same level,
# No changes to labels, losses or conditions of acceptance.
MAX_UPDATE_CANDIDATE_ATTEMPTS = 100
# formula(17)is every roundIF2FThe entrance of the route, the original implementation only uses an initial value.IPOPT
# Occasionally in this non-convexKKTThe problem converges to a locally infeasible point, so only in the standard
# precise+After all relaxation paths fail, add several preciseKKTMultiple initial values.
MAX_EQUATION17_EXACT_RESTARTS = 2
UPDATE_DUPLICATE_POINT_TOL = 1e-8
PREVIOUS_COMPATIBLE_METHOD_VERSION = (
    "paper_2d_kkt_v9_validated_cumulative_timing")
METHOD_VERSION = "paper_2d_kkt_v10_physical_sequence_stop"
INITIAL_DATA_METHOD_VERSION = "paper_uniform_balanced_slack_v5_timed"
LEGACY_INITIAL_DATA_METHOD_VERSION = "paper_uniform_iid_slack_v3"
# Reported offline time includes initial physical-label generation, so timed runs do not reuse an old label.
REUSE_LEGACY_INITIAL_DATA = False
LEGACY_INITIAL_DATA_ROOT = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "learning_methods" / CASENAME / "lin_paper_2d_formal" / "data")

FORMAL_CONDITIONS = int(shared.N_TEST)
STAGE_CONFIGS = {
    "diagnostic": {"n_conditions": 1, "initial_points": INITIAL_POINTS,
                   "update_rounds": 0, "update_points_per_route": 0,
                   "max_learning_steps": 200,
                   "output_name": "lin_paper_2d_diagnostic_validated_timed"},
    "formal": {"n_conditions": FORMAL_CONDITIONS,
               "initial_points": INITIAL_POINTS,
               "update_rounds": UPDATE_ROUNDS,
               "update_points_per_route": UPDATE_POINTS_PER_ROUTE,
               "max_learning_steps": MAX_LEARNING_STEPS,
               "output_name": "lin_paper_2d_formal_validated_timed"},
}

R16_TRUTH_PATH = shared.FORMAL_TEST_TRUTH_PATH

STAGE = "formal"
CONFIG = None
OUT = DATA_DIR = MODEL_DIR = EVAL_DIR = None


def training_cache_config():
    """Return every setting that changes the learned per-scenario polygon."""
    return {
        "method_version": METHOD_VERSION,
        "stage": STAGE,
        "n_sides": N_SIDES,
        "initial_points": CONFIG["initial_points"],
        "update_rounds": CONFIG["update_rounds"],
        "update_points_per_route": CONFIG["update_points_per_route"],
        "update_protocol": (
            "validated_misclassification_with_physical_sequence_stop_v2"),
        "duplicates_retained": True,
        "duplicate_point_tolerance": UPDATE_DUPLICATE_POINT_TOL,
        "max_update_candidate_attempts_per_route": (
            MAX_UPDATE_CANDIDATE_ATTEMPTS),
        "beta_fraction": BETA_FRACTION,
        "kkt_complementarity_tolerance": KKT_COMPLEMENTARITY_TOL,
        "kkt_relaxation_schedule": list(KKT_RELAXATION_SCHEDULE),
        "max_learning_steps": CONFIG["max_learning_steps"],
        "learning_rate": LEARNING_RATE,
        "learning_loss": "paper_equation_15_global_mean_hinge_v1",
        "score_scale_bounds": [SCORE_SCALE_MIN, SCORE_SCALE_MAX],
        "sampling_expansion_factor": SAMPLING_EXPANSION_FACTOR,
        "initial_sampling": "uniform_candidates_exact_class_balance_v5_timed",
        "cumulative_offline_timing": "checkpoint_safe_v1",
        "reuse_legacy_initial_labels": REUSE_LEGACY_INITIAL_DATA,
        "initial_feasible_target": CONFIG["initial_points"] // 2,
        "initial_infeasible_target": CONFIG["initial_points"] // 2,
        "feasibility_slack_tol": FEASIBILITY_SLACK_TOL,
        "polytope_classification_tol": POLYTOPE_CLASSIFICATION_TOL,
        "seed": SEED,
    }


def cache_matches_training_config(saved, dtheta, reference,
                                  require_complete=True):
    """Reject stale results when the condition or any training setting changed."""
    try:
        saved_config = json.loads(str(saved["training_config_json"]))
        saved_method = str(saved.get("method_version", ""))
        expected_config = training_cache_config()
        # v10 only changes what happens after a v9 round has exhausted all
        # equation-(23) candidates without completing the requested count.
        # Every round that v9 actually saved was generated with the same
        # equations, physical-label checks, loss, and accepted-point rules.
        # Therefore completed v9 rounds/results are a valid prefix/subset for
        # v10 and can be resumed without repeating expensive physical solves.
        if saved_method == PREVIOUS_COMPATIBLE_METHOD_VERSION:
            if saved_config.get("update_protocol") != (
                    "validated_misclassification_with_convergence_v1"):
                return False
            saved_config["method_version"] = METHOD_VERSION
            saved_config["update_protocol"] = (
                "validated_misclassification_with_physical_sequence_stop_v2")
            saved_method = METHOD_VERSION
        # The upper limit of candidate attempts is simply the computational budget before failure. change the upper limit from50Raise to100No
        # Alter any accepted points in already completed rounds, thus allowing oldv8breakpoint/Complete operating condition
        # Continue to use when the rest of the configuration is exactly the same and the new upper limit is not smaller.
        saved_attempt_cap = int(saved_config.pop(
            "max_update_candidate_attempts_per_route", -1))
        expected_attempt_cap = int(expected_config.pop(
            "max_update_candidate_attempts_per_route", -1))
        config_matches = (
            saved_config == expected_config
            and 0 < saved_attempt_cap <= expected_attempt_cap)
        method_matches = saved_method == METHOD_VERSION
        return (
            method_matches
            and (not require_complete or bool(saved.get("complete", False)))
            and config_matches
            and np.allclose(saved["dtheta"], dtheta, rtol=0.0, atol=1e-12)
            and np.allclose(saved["reference"], reference,
                            rtol=0.0, atol=1e-12)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def configure_stage(stage):
    global STAGE, CONFIG, OUT, DATA_DIR, MODEL_DIR, EVAL_DIR
    STAGE = stage
    CONFIG = dict(STAGE_CONFIGS[stage])
    OUT = (PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
           / "learning_methods" / CASENAME / CONFIG["output_name"])
    DATA_DIR, MODEL_DIR, EVAL_DIR = OUT / "data", OUT / "models", OUT / "evaluation"


def solver_succeeded(result):
    if result is None:
        return False
    return result.solver.termination_condition in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
    }


def solve_quietly(solver, model, failure_context=None):
    try:
        result = solver.solve(model, tee=False)
    except Exception as exc:
        if failure_context:
            print(f"[{failure_context}] Solver exception: {type(exc).__name__}: {exc}")
        return None
    if solver_succeeded(result):
        return result
    if failure_context:
        print(f"[{failure_context}] IPOPTUnsuccessful: status="
              f"{result.solver.status}, termination="
              f"{result.solver.termination_condition}, message="
              f"{result.solver.message}")
    return None


def deactivate_objectives(model):
    for objective in model.component_data_objects(pyo.Objective, active=True):
        objective.deactivate()


def update_parameters(ec, init_params, dtheta):
    shared.update_parameters(ec, init_params, dtheta)


def fixed_normals(n_sides=N_SIDES):
    angle = np.linspace(0.0, 2.0*np.pi, n_sides, endpoint=False)
    return np.column_stack((np.cos(angle), np.sin(angle)))


def normalize_halfspaces(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    norms = np.linalg.norm(A, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= ROW_NORM_TOL):
        raise RuntimeError("learnedAContains zero or non-limited rows.")
    A_normalized, b_normalized = A / norms[:, None], b / norms
    angles = np.sort(np.mod(np.arctan2(
        A_normalized[:, 1], A_normalized[:, 0]), 2.0*np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0*np.pi])
    if np.min(gaps) <= 1e-6:
        raise RuntimeError("learnedAContains duplicate normal rows.")
    if np.max(gaps) >= np.pi - 1e-8:
        raise RuntimeError("learnedAIt cannot be stretched into a two-dimensional space, and polygons may be unbounded..")
    return A_normalized, b_normalized


def halfspace_polygon_vertices(A, b, tolerance=1e-8):
    """Enumerate the vertices of a 2D half-empty polygon, used onlyKKTConstruct physics warm start after failure."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    candidates = []
    for first in range(len(A)):
        for second in range(first + 1, len(A)):
            matrix = np.stack((A[first], A[second]), axis=0)
            if abs(float(np.linalg.det(matrix))) <= 1e-12:
                continue
            point = np.linalg.solve(
                matrix, np.asarray([b[first], b[second]], dtype=float))
            if np.all(A @ point <= b + tolerance):
                candidates.append(point)
    if not candidates:
        return np.empty((0, 2), dtype=float)
    vertices = np.asarray(candidates, dtype=float)
    rounded = np.round(vertices, decimals=11)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return vertices[np.sort(unique_indices)]


def halfspaces_within_normal_sectors(A, atol=1e-6):
    """Check if each row is still in the equiangular sector assigned to it."""
    A = np.asarray(A, dtype=float)
    if A.shape != (N_SIDES, 2) or np.any(~np.isfinite(A)):
        return False
    norms = np.linalg.norm(A, axis=1)
    if np.any(norms <= ROW_NORM_TOL):
        return False
    angles = np.arctan2(A[:, 1], A[:, 0])
    centers = 2.0*np.pi*np.arange(N_SIDES) / N_SIDES
    delta = np.arctan2(np.sin(angles-centers), np.cos(angles-centers))
    half_width = NORMAL_SECTOR_FRACTION * 2.0*np.pi / N_SIDES
    return bool(np.all(np.abs(delta) <= half_width + atol))


def project_model_halfspaces(model):
    """Normalize each row and project the normals back to their respective sectors to ensure that the polygon normals are spannedR²."""
    with torch.no_grad():
        norms = torch.linalg.vector_norm(model.A, dim=1).clamp_min(
            ROW_NORM_TOL)
        # A,bWhen dividing by the row norm at the same time, the geometry remains unchanged; transfer the norm to the positive scale,
        # make thesishingeThe fraction also remains the same before and after projection.
        model.log_scales.add_(torch.log(norms))
        model.log_scales.clamp_(
            min=float(np.log(SCORE_SCALE_MIN)),
            max=float(np.log(SCORE_SCALE_MAX)))
        model.b.div_(norms)
        unit_A = model.A / norms[:, None]
        angles = torch.atan2(unit_A[:, 1], unit_A[:, 0])
        centers = (2.0*torch.pi*torch.arange(
            N_SIDES, device=model.A.device, dtype=model.A.dtype) / N_SIDES)
        delta = torch.atan2(torch.sin(angles-centers),
                            torch.cos(angles-centers))
        half_width = NORMAL_SECTOR_FRACTION * 2.0*torch.pi / N_SIDES
        angles = centers + torch.clamp(delta, -half_width, half_width)
        model.A[:, 0].copy_(torch.cos(angles))
        model.A[:, 1].copy_(torch.sin(angles))


class PaperPolytope(nn.Module):
    """Paper parameters for a single fixed operating conditionA,b; Nothetainput."""

    def __init__(self, A_init, b_init, score_scale_init=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.as_tensor(A_init, dtype=torch.float32))
        self.b = nn.Parameter(torch.as_tensor(b_init, dtype=torch.float32))
        scale = float(np.clip(
            score_scale_init, SCORE_SCALE_MIN, SCORE_SCALE_MAX))
        self.log_scales = nn.Parameter(torch.full(
            (len(A_init),), float(np.log(scale)), dtype=torch.float32))

    def violation_score(self, points):
        # Positive scaling preserves the region Ax <= b and restores the normalization used in Eq. (15).
        # Row scale degrees of freedom; geometric normal vectors remain unitized.
        scales = torch.exp(self.log_scales)
        return torch.max(
            (points @ self.A.T - self.b) * scales[None, :], dim=1).values


class ExactScenarioPolytopeBank(nn.Module):
    """Only allows precise search of trained conditions; no newthetaDo nearest neighbor generalization."""

    def __init__(self, conditions, A_all, b_all, match_tol=1e-6):
        super().__init__()
        self.match_tol = float(match_tol)
        self.register_buffer("conditions", torch.as_tensor(conditions, dtype=torch.float32))
        self.register_buffer("A_all", torch.as_tensor(A_all, dtype=torch.float32))
        self.register_buffer("b_all", torch.as_tensor(b_all, dtype=torch.float32))

    def forward(self, theta):
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        error = torch.amax(torch.abs(
            theta[:, None, :] - self.conditions[None, :, :]), dim=2)
        value, index = torch.min(error, dim=1)
        if bool(torch.any(value > self.match_tol)):
            raise ValueError("LinThe polygon library receives new cases that were not independently trained.")
        return self.A_all[index], self.b_all[index]


def paper_hinge_loss(model, points, labels):
    """Essay style(15): Average all data points in the training set with equal weights.."""
    losses = F.relu(1.0 - labels * model.violation_score(points))
    if not bool(torch.any(labels < 0)) or not bool(torch.any(labels > 0)):
        raise RuntimeError("The paper learning loss needs to contain both feasible and infeasible labels.")
    return torch.mean(losses)


def fit_learning_module(model, points, labels, stage_name):
    """Only optimize the formula according to a predetermined number of steps(15), Do not select weights with true bounds or training loss."""
    points_t = torch.as_tensor(points, dtype=torch.float32, device=DEVICE)
    labels_t = torch.as_tensor(labels, dtype=torch.float32, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    started = time.perf_counter()
    for step in range(1, CONFIG["max_learning_steps"] + 1):
        optimizer.zero_grad()
        loss = paper_hinge_loss(model, points_t, labels_t)
        loss.backward()
        optimizer.step()
        # The row scale itself does not represent geometric information; unitizing and limiting the angular sector of each row allows both
    # Learning edge directions prevents normals from collapsing and producing an unbounded polygon.
        project_model_halfspaces(model)
        if step == 1 or step % LOSS_CHECK_INTERVAL == 0:
            value = float(loss.detach())
            history.append((step, value))
            print(f"[{stage_name}] step={step}, paper_loss={value:.6e}")
    return np.asarray(history, dtype=float), time.perf_counter() - started


def load_common_conditions(dim_theta):
    dtheta = load_common_test_conditions(dim_theta, overwrite=False)
    if len(dtheta) < CONFIG["n_conditions"]:
        raise ValueError(f"Public files only{len(dtheta)}test conditions, currently required"
                         f"{CONFIG['n_conditions']}a.")
    return dtheta[:CONFIG["n_conditions"]]


def load_or_prepare_common_truth(case, ppc, conditions, overwrite=False):
    if STAGE == "formal":
        truth_path = R16_TRUTH_PATH
    else:
        truth_path = EVAL_DIR / "common_truth_subset.npz"
    if STAGE == "formal" and truth_path.exists():
        with np.load(truth_path) as truth:
            if (len(truth["dtheta"]) == len(conditions)
                    and np.allclose(truth["dtheta"], conditions,
                                    rtol=0.0, atol=1e-7)):
                return truth_path
    return shared.prepare_truth(
        case, ppc, len(conditions), shared.N_ERROR_DIRECTIONS,
        shared.N_COVERAGE_DIRECTIONS, overwrite=overwrite,
        truth_path=truth_path, theta_override=conditions)


def formal_truth_cache_matches(conditions):
    """Read-only validation used before concurrently starting train shards."""
    if not R16_TRUTH_PATH.exists():
        return False
    try:
        with np.load(R16_TRUTH_PATH) as truth:
            return (
                truth["dtheta"].shape == conditions.shape
                and np.allclose(truth["dtheta"], conditions,
                                rtol=0.0, atol=1e-7)
                and truth["error_directions"].shape[1]
                    == shared.N_ERROR_DIRECTIONS
                and truth["coverage_directions"].shape[1]
                    == shared.N_COVERAGE_DIRECTIONS
                and str(truth.get("coverage_truth_definition", ""))
                    == "max_k_ray_model_v1"
            )
    except (KeyError, OSError, ValueError):
        return False


def device_tdi_sampling_box(ppc, init_params, dtheta, reference):
    """Build a sampling box around the no-flexibility power-flow point using the physical adjustment limits."""
    pd_meta0 = np.asarray(init_params["Pd_meta"]["initial_value"], dtype=float)
    qd_meta0 = np.asarray(init_params["Qd_meta"]["initial_value"], dtype=float)
    n = pd_meta0.size
    pd_meta = pd_meta0 + np.asarray(dtheta[:n]).reshape(pd_meta0.shape)
    qd_meta = qd_meta0 + np.asarray(dtheta[n:]).reshape(qd_meta0.shape)
    base = float(ppc["baseMVA"])
    pd_nominal = np.asarray(ppc["bus"][:, 2], dtype=float) / base
    qd_nominal = np.asarray(ppc["bus"][:, 3], dtype=float) / base
    pd = (0.8 + 0.4*np.asarray(pd_meta).reshape(-1)) * pd_nominal
    qd = (0.8 + 0.4*np.asarray(qd_meta).reshape(-1)) * qd_nominal
    flex = ppc["node_flex_dict"]
    radius_p = radius_q = 0.0
    for index, row in enumerate(ppc["bus"]):
        info = flex[int(row[0])]
        if info["type"] == 1:
            radius_p += float(info["rate"]) * abs(pd[index])
        elif info["type"] == 2:
            radius_p += float(info["rate"][0]) * abs(pd[index])
            radius_q += float(info["rate"][1]) * abs(qd[index])
        elif info["type"] == 3:
            radius = np.sqrt(float(info["rate"])
                             * (pd[index]**2 + qd[index]**2))
            radius_p += radius
            radius_q += radius
    radii = SAMPLING_EXPANSION_FACTOR * np.maximum(
        [radius_p, radius_q], MIN_AXIS_RADIUS)
    reference = np.asarray(reference, dtype=float)
    return reference - radii, reference + radii, radii


def build_relaxed_physics_model(physical_model, fixed_target=False,
                                violation_floor=False):
    """Construct a thesis format(17)Inner layer: Equality maintained, all running upper and lower bounds relaxed with non-negative relaxation."""
    model = physical_model.clone()
    deactivate_objectives(model)
    original_constraints = list(model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True))
    original_variables = list(model.component_data_objects(
        pyo.Var, active=True, descend_into=True))

    variable_bounds = []
    for variable in original_variables:
        if variable.fixed:
            continue
        lower, upper = variable.lb, variable.ub
        if lower is not None or upper is not None:
            variable_bounds.append((variable, lower, upper))
            variable.setlb(None)
            variable.setub(None)
            variable.domain = pyo.Reals

    model.paper_slacks = pyo.VarList(domain=pyo.Reals)
    model.paper_slack_nonnegative = pyo.ConstraintList()
    model.paper_relaxed_constraints = pyo.ConstraintList()
    def add_slack():
        slack = model.paper_slacks.add()
        model.paper_slack_nonnegative.add(slack >= 0.0)
        return slack
    for constraint in original_constraints:
        if constraint.equality:
            continue
        constraint.deactivate()
        if constraint.lower is not None:
            slack = add_slack()
            model.paper_relaxed_constraints.add(
                constraint.body >= constraint.lower - slack)
        if constraint.upper is not None:
            slack = add_slack()
            model.paper_relaxed_constraints.add(
                constraint.body <= constraint.upper + slack)
    for variable, lower, upper in variable_bounds:
        if lower is not None:
            slack = add_slack()
            model.paper_relaxed_constraints.add(variable >= lower - slack)
        if upper is not None:
            slack = add_slack()
            model.paper_relaxed_constraints.add(variable <= upper + slack)

    model.paper_total_slack = pyo.Expression(
        expr=sum(model.paper_slacks[index] for index in model.paper_slacks))
    if fixed_target:
        model.paper_target = pyo.Param(range(2), mutable=True, initialize=0.0)
        model.paper_target_constraints = pyo.ConstraintList()
        for j in range(2):
            model.paper_target_constraints.add(
                model.var_proj[j] == model.paper_target[j])
    if violation_floor:
        model.paper_violation_floor = pyo.Param(mutable=True, initialize=0.0)
        model.paper_floor_constraint = pyo.Constraint(
            expr=model.paper_total_slack >= model.paper_violation_floor)
    return model


class FeasibilitySlackOracle:
    """Unify initial point and update point labels: find the minimum total relaxation of the paperphi(x)."""

    def __init__(self, physical_model):
        self.model = build_relaxed_physics_model(physical_model, fixed_target=True)
        # IPOPTReturns the dual multiplier of the inner minimum relaxation problem, given by(17)
        # Use a primal-dual-consistent warm start if the standard KKT initialization fails.
        self.model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        self.model.paper_inner_objective = pyo.Objective(
            expr=self.model.paper_total_slack, sense=pyo.minimize)
        self.solver = pyo.SolverFactory("ipopt", tee=False)
        self.solver.options["tol"] = 1e-9
        self.solver.options["max_iter"] = 10000

    def evaluate(self, point):
        for j in range(2):
            self.model.paper_target[j] = float(point[j])
        result = solve_quietly(self.solver, self.model)
        if result is None:
            return None
        value = float(pyo.value(self.model.paper_total_slack))
        return value if np.isfinite(value) else None

    def variable_values(self):
        """Save the latest inner minimum relaxation solution for isomorphismKKTModel hot start."""
        return {
            variable.name: float(variable.value)
            for variable in self.model.component_data_objects(
                pyo.Var, active=True, descend_into=True)
            if variable.value is not None and np.isfinite(variable.value)
        }

    def kkt_warm_start_data(self):
        """putPyomoDual symbols are converted to this articleg<=0ofKKTmultiplier agreement."""
        equality_multipliers = {}
        inequality_multipliers = {}
        for constraint in self.model.component_data_objects(
                pyo.Constraint, active=True, descend_into=True):
            dual = self.model.dual.get(constraint)
            if dual is None or not np.isfinite(dual):
                continue
            dual = float(dual)
            if constraint.equality:
                # Pyomo/Ipoptthe equation ofdualwith L=f+lambda(body-rhs)
                # inlambdaopposite sign.
                equality_multipliers[(constraint.name, "equality")] = -dual
            else:
                if constraint.lower is not None:
                    inequality_multipliers[(constraint.name, "lower")] = max(
                        dual, 0.0)
                if constraint.upper is not None:
                    inequality_multipliers[(constraint.name, "upper")] = max(
                        -dual, 0.0)
        return {
            "variable_values": self.variable_values(),
            "equality_multipliers": equality_multipliers,
            "inequality_multipliers": inequality_multipliers,
        }


def canonical_constraint_expressions(model):
    """Write the activity constraints ash=0org<=0, supplyKKTLagrangian function use."""
    equalities, inequalities = [], []
    equality_keys, inequality_keys = [], []
    for constraint in model.component_data_objects(
            pyo.Constraint, active=True, descend_into=True):
        if constraint.equality:
            equalities.append(constraint.body - constraint.lower)
            equality_keys.append((constraint.name, "equality"))
        else:
            if constraint.lower is not None:
                inequalities.append(constraint.lower - constraint.body)
                inequality_keys.append((constraint.name, "lower"))
            if constraint.upper is not None:
                inequalities.append(constraint.body - constraint.upper)
                inequality_keys.append((constraint.name, "upper"))
    return equalities, inequalities, equality_keys, inequality_keys


def add_inner_kkt_conditions(model):
    """Replace the minimum total slack inner problem withKKTnecessary conditions."""
    outer_variable_ids = {id(model.var_proj[j]) for j in range(2)}
    inner_variables = [
        variable for variable in model.component_data_objects(
            pyo.Var, active=True, descend_into=True)
        if not variable.fixed and id(variable) not in outer_variable_ids
    ]
    (equalities, inequalities,
     equality_keys, inequality_keys) = canonical_constraint_expressions(model)
    model.paper_lambda = pyo.VarList(domain=pyo.Reals)
    model.paper_mu = pyo.VarList(domain=pyo.NonNegativeReals)
    lambdas = [model.paper_lambda.add() for _ in equalities]
    multipliers = [model.paper_mu.add() for _ in inequalities]
    # The stationary form of the relaxation variable contains a constant gradient1; If all inequality multipliers start fromPyomodefault0
    # Starting, the initial point will seriously violate the stationary point condition., IPOPTIt is easy to converge to “locally infeasible”.
    # mu = 1 matches the scale of the minimum-total-relaxation problem and only initializes the KKT solve.
    for multiplier in multipliers:
        multiplier.set_value(1.0)
    lagrangian = model.paper_total_slack
    lagrangian += sum(multiplier * expression
                      for multiplier, expression in zip(lambdas, equalities))
    lagrangian += sum(multiplier * expression
                      for multiplier, expression in zip(multipliers, inequalities))
    model.paper_stationarity = pyo.ConstraintList()
    for variable in inner_variables:
        derivative = differentiate(
            lagrangian, wrt=variable, mode=Modes.reverse_symbolic)
        model.paper_stationarity.add(derivative == 0.0)
    # tau = 0 gives exact complementarity; tau > 0 is only a fallback for initialization.
    # canonical_constraint_expressionsUse uniformlyg<=0, Therefore-mu*g>=0.
    model.paper_complementarity_tolerance = pyo.Param(
        mutable=True, initialize=0.0)
    model.paper_complementarity = pyo.ConstraintList()
    for multiplier, expression in zip(multipliers, inequalities):
        model.paper_complementarity.add(
            -multiplier * expression
            <= model.paper_complementarity_tolerance)
    model.paper_kkt_multipliers = tuple(multipliers)
    model.paper_kkt_lambdas = tuple(lambdas)
    model.paper_kkt_equality_keys = tuple(equality_keys)
    model.paper_kkt_inequality_keys = tuple(inequality_keys)
    model.paper_kkt_inequality_expressions = tuple(inequalities)
    return len(inner_variables), len(equalities), len(inequalities)


class KKTOvercoverageUpdater:
    """Essay style(17)/(23): Maximum true operating constraint relaxation within the prediction domain."""

    def __init__(self, physical_model, with_floor=False):
        self.model = build_relaxed_physics_model(
            physical_model, fixed_target=False, violation_floor=with_floor)
        self.with_floor = with_floor
        self.model.paper_A = pyo.Param(
            range(N_SIDES), range(2), mutable=True, initialize=0.0)
        self.model.paper_b = pyo.Param(
            range(N_SIDES), mutable=True, initialize=0.0)
        self.model.paper_predicted_region = pyo.ConstraintList()
        for i in range(N_SIDES):
            self.model.paper_predicted_region.add(
                sum(self.model.paper_A[i, j] * self.model.var_proj[j]
                    for j in range(2)) <= self.model.paper_b[i])
    # Disable the outer prediction polyhedron before constructing the inner-layer KKT system in Eq. (24).
        # Explicitly add the relaxed lower boundU(j), Sowith_floorit remains in the inner layer.
        predicted = list(self.model.paper_predicted_region.values())
        for constraint in predicted:
            constraint.deactivate()
        self.kkt_size = add_inner_kkt_conditions(self.model)
        for constraint in predicted:
            constraint.activate()
        self.model.paper_outer_objective = pyo.Objective(
            expr=self.model.paper_total_slack, sense=pyo.maximize)
        self.solver = pyo.SolverFactory("ipopt", tee=False)
        self.solver.options["tol"] = 1e-8
        self.solver.options["max_iter"] = 20000
        # A very small number of bad initial values will makeIPOPTstuck in recovery phase for a long time; after timeout
        # Advance to the next controlled warm start so a stalled shard cannot block indefinitely.
        self.solver.options["max_cpu_time"] = 120.0
        self.solver.options["mu_strategy"] = "adaptive"
        self._last_A = None
        self._last_b = None
        self._last_floor = None
        self._last_success_values = None
        # Keep the clean initial values when the model is constructed. failedIPOPTRunning will rewrite
        # PyomoVariable value, cannot allow the next “restart“ to continue from the point of failure.
        self._cold_start_values = self._snapshot_values()
        self.last_solve_metadata = {}

    def set_polytope(self, A, b):
        for i in range(N_SIDES):
            for j in range(2):
                self.model.paper_A[i, j] = float(A[i, j])
            self.model.paper_b[i] = float(b[i])

    def _snapshot_values(self):
        return [variable.value for variable in
                self.model.component_data_objects(
                    pyo.Var, active=True, descend_into=True)]

    def _restore_values(self, values):
        variables = list(self.model.component_data_objects(
            pyo.Var, active=True, descend_into=True))
        if values is None or len(values) != len(variables):
            return
        for variable, value in zip(variables, values):
            if value is not None and np.isfinite(value):
                variable.set_value(float(value), skip_validation=True)

    def _jitter_values(self, values, seed):
        """Provide different initial values for the same recursion level to avoid returning to the same part repeatedlyKKTpoint."""
        rng = np.random.default_rng(int(seed))
        variables = list(self.model.component_data_objects(
            pyo.Var, active=True, descend_into=True))
        jittered = []
        for variable, value in zip(variables, values):
            if value is None or not np.isfinite(value):
                jittered.append(value)
                continue
            value = float(value)
            lower = (None if variable.lb is None
                     else float(pyo.value(variable.lb)))
            upper = (None if variable.ub is None
                     else float(pyo.value(variable.ub)))
            if lower is not None and upper is not None and upper > lower:
                scale = max(0.02 * (upper-lower), 1e-6)
            else:
                scale = max(0.05 * abs(value), 1e-4)
            trial = value + float(rng.normal(0.0, scale))
            if lower is not None:
                trial = max(trial, lower)
            if upper is not None:
                trial = min(trial, upper)
            jittered.append(trial)
        return jittered

    def set_physical_warm_start(self, warm_start_data):
        """Use fixedTDIinner layer-Dual solution initialization formula(17) KKT."""
        self._restore_values(self._cold_start_values)
        source_values = warm_start_data["variable_values"]
        for variable in self.model.component_data_objects(
                pyo.Var, active=True, descend_into=True):
            value = source_values.get(variable.name)
            if value is not None and np.isfinite(value):
                variable.set_value(float(value), skip_validation=True)
        equality_multipliers = warm_start_data["equality_multipliers"]
        inequality_multipliers = warm_start_data["inequality_multipliers"]
        for multiplier, key in zip(
                self.model.paper_kkt_lambdas,
                self.model.paper_kkt_equality_keys):
            multiplier.set_value(
                float(equality_multipliers.get(key, 0.0)),
                skip_validation=True)
        for multiplier, key in zip(
                self.model.paper_kkt_multipliers,
                self.model.paper_kkt_inequality_keys):
            multiplier.set_value(
                float(inequality_multipliers.get(key, 0.0)),
                skip_validation=True)
        self._last_success_values = None
        self._cold_start_values = self._snapshot_values()

    def _set_problem_parameters(self, A, b, floor, tau):
        self.set_polytope(A, b)
        if self.with_floor:
            self.model.paper_violation_floor = max(float(floor), 0.0)
        self.model.paper_complementarity_tolerance = max(float(tau), 0.0)

    def _exact_complementarity_residual(self):
        values = [
            abs(float(pyo.value(multiplier * expression)))
            for multiplier, expression in zip(
                self.model.paper_kkt_multipliers,
                self.model.paper_kkt_inequality_expressions)
        ]
        return max(values or [0.0])

    def _try_stage(self, A, b, floor, tau, context):
        self._set_problem_parameters(A, b, floor, tau)
        result = solve_quietly(self.solver, self.model, context)
        if result is None:
            return False
        point = np.asarray([pyo.value(self.model.var_proj[j])
                            for j in range(2)], dtype=float)
        violation = float(pyo.value(self.model.paper_total_slack))
        residual = self._exact_complementarity_residual()
        finite = (np.all(np.isfinite(point)) and np.isfinite(violation)
                  and np.isfinite(residual))
        if not finite:
            print(f"[{context}] There are non-finite values in the solution.")
            return False
        # tau=0is a formally acceptable paperKKTsolution.
        if tau == 0.0 and residual > KKT_COMPLEMENTARITY_TOL:
            print(f"[{context}] preciseKKTcomplementary residuals{residual:.3e}exceed"
                  f"{KKT_COMPLEMENTARITY_TOL:.1e}.")
            return False
        return True

    def _solve_relaxation_path(self, A, b, floor, prefix):
        stages = []
        for tau in KKT_RELAXATION_SCHEDULE:
            context = f"{prefix}/tau={tau:.0e}"
            if not self._try_stage(A, b, floor, tau, context):
                return False, stages
            stages.append(float(tau))
        return True, stages

    def solve(self, A, b, floor=None, restart_seed=None, commit=True,
              allow_fallback=True):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        floor = 0.0 if floor is None else max(float(floor), 0.0)
        starting_values = (self._last_success_values
                           if self._last_success_values is not None
                           else self._cold_start_values)
        if restart_seed is not None:
            starting_values = self._jitter_values(
                starting_values, restart_seed)
        attempts = []

        # Normal path strictly solves the paperKKT; Most rounds require no additional overhead.
        self._restore_values(starting_values)
        exact_ok = self._try_stage(
            A, b, floor, 0.0, "KKT/exact")
        attempts.append({"kind": "exact", "success": bool(exact_ok)})

        if not exact_ok and allow_fallback:
            # First in the targetA,bgradually tightens the complementary slack, and eventually returns totau=0.
            self._restore_values(starting_values)
            relaxed_ok, stages = self._solve_relaxation_path(
                A, b, floor, "KKT/relax")
            attempts.append({"kind": "relaxation", "stages": stages,
                             "success": bool(relaxed_ok)})
            exact_ok = relaxed_ok

        if (not exact_ok and allow_fallback and self._last_A is not None
                and self._last_b is not None):
        # Continue from the last successful polygon when active sets jump after facet reordering.
            self._restore_values(starting_values)
            continuation_ok = True
            completed = []
            old_floor = (floor if self._last_floor is None
                         else float(self._last_floor))
            for fraction in KKT_POLYTOPE_CONTINUATION:
                A_step = ((1.0-fraction)*self._last_A + fraction*A)
                b_step = ((1.0-fraction)*self._last_b + fraction*b)
                floor_step = (1.0-fraction)*old_floor + fraction*floor
                ok, stages = self._solve_relaxation_path(
                    A_step, b_step, floor_step,
                    f"KKT/continuation={fraction:.2f}")
                completed.append({"fraction": float(fraction),
                                  "stages": stages,
                                  "success": bool(ok)})
                if not ok:
                    continuation_ok = False
                    break
            attempts.append({"kind": "polytope_continuation",
                             "steps": completed,
                             "success": bool(continuation_ok)})
            exact_ok = continuation_ok

        if not exact_ok:
            self.last_solve_metadata = {
                "success": False, "attempts": attempts,
                "final_complementarity_tolerance": None,
            }
            return None

        point = np.asarray([pyo.value(self.model.var_proj[j])
                            for j in range(2)], dtype=float)
        violation = float(pyo.value(self.model.paper_total_slack))
        complementarity = self._exact_complementarity_residual()
        if not np.all(np.isfinite(point)) or not np.isfinite(violation):
            return None
        if commit:
            self.commit_current_solution(A, b, floor)
        self.last_solve_metadata = {
            "success": True,
            "attempts": attempts,
            "final_complementarity_tolerance": 0.0,
            "exact_complementarity_residual": float(complementarity),
        }
        return point, violation, complementarity

    def commit_current_solution(self, A, b, floor=None):
        """Only submitted as a warm start for the next solution if the candidate has passed independent physics review."""
        self._last_A = np.asarray(A, dtype=float).copy()
        self._last_b = np.asarray(b, dtype=float).copy()
        self._last_floor = 0.0 if floor is None else float(floor)
        self._last_success_values = self._snapshot_values()


class TrueDomainHyperplaneUpdater:
    """Essay style(21)/(25)-(27): The maximum violation of the predicted hyperplane by the true feasible region."""

    def __init__(self, physical_model):
        self.model = physical_model.clone()
        deactivate_objectives(self.model)
        self.model.paper_a = pyo.Param(range(2), mutable=True, initialize=0.0)
        self.model.paper_b = pyo.Param(mutable=True, initialize=0.0)
        self.model.paper_cap = pyo.Param(mutable=True, initialize=1e6)
        expression = (sum(self.model.paper_a[j] * self.model.var_proj[j]
                          for j in range(2)) - self.model.paper_b)
        self.model.paper_violation = pyo.Expression(expr=expression)
        self.model.paper_cap_constraint = pyo.Constraint(
            expr=self.model.paper_violation <= self.model.paper_cap)
        self.model.paper_cap_constraint.deactivate()
        self.model.paper_objective = pyo.Objective(
            expr=self.model.paper_violation, sense=pyo.maximize)
        self.solver = pyo.SolverFactory("ipopt", tee=False)
        self.solver.options["tol"] = 1e-9
        self.solver.options["max_iter"] = 10000

    def _jitter_start(self, seed):
        rng = np.random.default_rng(int(seed))
        for variable in self.model.component_data_objects(
                pyo.Var, active=True, descend_into=True):
            if variable.fixed or variable.value is None:
                continue
            value = float(variable.value)
            lower = (None if variable.lb is None
                     else float(pyo.value(variable.lb)))
            upper = (None if variable.ub is None
                     else float(pyo.value(variable.ub)))
            if lower is not None and upper is not None and upper > lower:
                scale = max(0.02 * (upper-lower), 1e-6)
            else:
                scale = max(0.05 * abs(value), 1e-4)
            trial = value + float(rng.normal(0.0, scale))
            if lower is not None:
                trial = max(trial, lower)
            if upper is not None:
                trial = min(trial, upper)
            variable.set_value(trial, skip_validation=True)

    def solve(self, a, b, cap=None, restart_seed=None):
        for j in range(2):
            self.model.paper_a[j] = float(a[j])
        self.model.paper_b = float(b)
        if cap is None:
            self.model.paper_cap_constraint.deactivate()
        else:
            self.model.paper_cap = float(cap)
            self.model.paper_cap_constraint.activate()
        if restart_seed is not None:
            self._jitter_start(restart_seed)
        result = solve_quietly(self.solver, self.model)
        if result is None:
            return None
        point = np.asarray([pyo.value(self.model.var_proj[j])
                            for j in range(2)], dtype=float)
        violation = float(pyo.value(self.model.paper_violation))
        return (point, violation) if np.all(np.isfinite(point)) else None


def prepare_condition_context(case, ppc, dtheta, reference):
    ec = case["errorcalculator"].copy()
    ec.solver.options["tol"] = 1e-9
    ec.solver.options["max_iter"] = 10000
    update_parameters(ec, case["params"]["params_dict"], dtheta)
    lower, upper, radii = device_tdi_sampling_box(
        ppc, case["params"]["params_dict"], dtheta, reference)
    return ec, lower, upper, radii


def save_initial_data(path, points, labels, violations, attempts,
                      candidate_feasible, candidate_infeasible,
                      solver_failures, lower, upper, metadata,
                      generation_seconds=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, points=np.asarray(points, dtype=float),
             labels=np.asarray(labels, dtype=np.int8),
             physical_slack_violation=np.asarray(violations, dtype=float),
             attempts=np.asarray(attempts),
             candidate_feasible=np.asarray(candidate_feasible),
             candidate_infeasible=np.asarray(candidate_infeasible),
             solver_failures=np.asarray(solver_failures),
             sampling_lower=np.asarray(lower, dtype=float),
             sampling_upper=np.asarray(upper, dtype=float),
             sampling_expansion_factor=np.asarray(SAMPLING_EXPANSION_FACTOR),
             generation_seconds=np.asarray(float(generation_seconds)),
             metadata_json=np.asarray(json.dumps(metadata)),
             method_version=np.asarray(INITIAL_DATA_METHOD_VERSION))


def generate_initial_data(ci, oracle, lower, upper, overwrite=False):
    """Initial set of papers: It is feasible to generate candidates uniformly and select half of them respectively./infeasible sample."""
    path = DATA_DIR / f"scenario_{ci:03d}" / "initial_points.npz"
    reused_initial_data_from = None
    previous_generation_seconds = 0.0
    # The method changes only the update protocol, learning loss, initial label, and seed.
    # Operating conditions and sampling bounds remain unchanged. Reused labels are
    # validated against the method version, bounds, and class count; old results stay read-only.
    if REUSE_LEGACY_INITIAL_DATA and not path.exists() and not overwrite:
        legacy_path = (
            LEGACY_INITIAL_DATA_ROOT / f"scenario_{ci:03d}"
            / "initial_points.npz")
        if legacy_path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, path)
            reused_initial_data_from = str(legacy_path)
            print(f"[scenario {ci:03d}/data] Copy the old experiment initial label and verify it strictly: "
                  f"{legacy_path}")
    target_total = CONFIG["initial_points"]
    if target_total % 2:
        raise ValueError("Category balancing requirementsinitial_pointsis an even number.")
    target_each = target_total // 2
    points, labels, violations, attempts = [], [], [], 0
    candidate_feasible = candidate_infeasible = failures = 0
    if path.exists() and not overwrite:
        with np.load(path) as old:
            old_method = str(old.get("method_version", ""))
            try:
                old_metadata = json.loads(str(old.get("metadata_json", "{}")))
            except (TypeError, ValueError, json.JSONDecodeError):
                old_metadata = {}
            reused_initial_data_from = (
                reused_initial_data_from
                or old_metadata.get("reused_initial_data_from"))
            previous_generation_seconds = float(
                old.get("generation_seconds", 0.0))
            geometry_matches = (
                float(old.get("sampling_expansion_factor", np.nan))
                == SAMPLING_EXPANSION_FACTOR
                and np.allclose(old.get("sampling_lower", []), lower,
                                rtol=0.0, atol=1e-12)
                and np.allclose(old.get("sampling_upper", []), upper,
                                rtol=0.0, atol=1e-12)
            )
            if (old_method in {
                    INITIAL_DATA_METHOD_VERSION,
                    LEGACY_INITIAL_DATA_METHOD_VERSION}
                    and geometry_matches):
                old_points = np.asarray(old["points"], dtype=float)
                old_labels = np.asarray(old["labels"], dtype=int)
                old_violations = np.asarray(
                    old["physical_slack_violation"], dtype=float)
                # The old version was saved before10000success label. Filter various categories according to the original generation order
                # 5000, exactly the same as performing class quota sampling from the same random stream.
                keep = []
                kept_feasible = kept_infeasible = 0
                for index, label in enumerate(old_labels):
                    if label < 0 and kept_feasible < target_each:
                        keep.append(index)
                        kept_feasible += 1
                    elif label > 0 and kept_infeasible < target_each:
                        keep.append(index)
                        kept_infeasible += 1
                points = list(old_points[keep])
                labels = list(old_labels[keep])
                violations = list(old_violations[keep])
                attempts = int(old["attempts"])
                candidate_feasible = int(old["candidate_feasible"])
                candidate_infeasible = int(old["candidate_infeasible"])
                failures = int(old["solver_failures"])
                if old_method == LEGACY_INITIAL_DATA_METHOD_VERSION:
                    print(f"[scenario {ci:03d}/data] Reuse old uniform candidate labels: "
                          f"Reserved available{kept_feasible}/{target_each}, "
                          f"Not feasible{kept_infeasible}/{target_each}, "
                          "Continue to fill in missing categories.")
                if (kept_feasible == target_each
                        and kept_infeasible == target_each):
                    save_initial_data(
                        path, points, labels, violations, attempts,
                        candidate_feasible, candidate_infeasible, failures,
                        lower, upper, {
                            "scenario": ci,
                            "balanced_feasible_target": target_each,
                            "balanced_infeasible_target": target_each,
                            "migrated_from": old_method,
                            "reused_initial_data_from": (
                                reused_initial_data_from),
                        }, generation_seconds=previous_generation_seconds)
                    report_sampling_diagnostic(
                        ci, path, attempts, candidate_feasible,
                        candidate_infeasible, failures,
                        kept_feasible, kept_infeasible)
                    return path
    rng = np.random.RandomState(SEED + 10_000 + ci)
    if attempts:
        rng.uniform(lower, upper, size=(attempts, 2))
    max_attempts = MAX_SAMPLE_ATTEMPT_FACTOR * CONFIG["initial_points"]
    started = time.perf_counter()

    def cumulative_generation_seconds():
        return previous_generation_seconds + time.perf_counter() - started

    last_save_count = len(labels)
    accepted_feasible = int(np.count_nonzero(np.asarray(labels) < 0))
    accepted_infeasible = int(np.count_nonzero(np.asarray(labels) > 0))
    while (accepted_feasible < target_each
           or accepted_infeasible < target_each):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Working conditions{ci}Unable to collect full equilibrium initial set: feasible="
                f"{accepted_feasible}/{target_each}, Not feasible="
                f"{accepted_infeasible}/{target_each}.")
        point = rng.uniform(lower, upper)
        violation = oracle.evaluate(point)
        if violation is None:
            failures += 1
            continue
        feasible = violation <= FEASIBILITY_SLACK_TOL
        if feasible:
            candidate_feasible += 1
            if accepted_feasible >= target_each:
                continue
            accepted_feasible += 1
        else:
            candidate_infeasible += 1
            if accepted_infeasible >= target_each:
                continue
            accepted_infeasible += 1
        points.append(point)
        labels.append(-1 if feasible else 1)
        violations.append(violation)
        if len(labels) - last_save_count >= 100:
            save_initial_data(
                path, points, labels, violations, attempts,
                candidate_feasible, candidate_infeasible, failures,
                lower, upper, {
                "scenario": ci, "lower": lower.tolist(), "upper": upper.tolist(),
                "solver_failures": failures,
                "balanced_feasible_target": target_each,
                "balanced_infeasible_target": target_each,
                "elapsed_current_run": time.perf_counter() - started},
                generation_seconds=cumulative_generation_seconds())
            last_save_count = len(labels)
            print(f"[scenario {ci:03d}/data] {len(labels)}/"
                  f"{CONFIG['initial_points']}, attempts={attempts}")
    save_initial_data(
        path, points, labels, violations, attempts,
        candidate_feasible, candidate_infeasible, failures,
        lower, upper, {
        "scenario": ci, "lower": lower.tolist(), "upper": upper.tolist(),
        "solver_failures": failures,
        "balanced_feasible_target": target_each,
        "balanced_infeasible_target": target_each,
        "elapsed_current_run": time.perf_counter() - started},
        generation_seconds=cumulative_generation_seconds())
    report_sampling_diagnostic(
        ci, path, attempts, candidate_feasible, candidate_infeasible, failures,
        accepted_feasible, accepted_infeasible)
    return path


def report_sampling_diagnostic(ci, data_path, attempts, candidate_feasible,
                               candidate_infeasible, solver_failures,
                               accepted_feasible, accepted_infeasible):
    """Save categories for diagnosis; only report suggestions, do not automatically adjust sampling boxes in formal experiments."""
    solved = candidate_feasible + candidate_infeasible
    feasible_rate = candidate_feasible / solved if solved else np.nan
    threshold = SAMPLING_CLASS_SCARCITY_THRESHOLD
    if not np.isfinite(feasible_rate):
        recommendation = "solver produced no classifiable candidates"
    elif feasible_rate > 1.0 - threshold:
        recommendation = "infeasible candidates scarce: test a larger factor"
    elif feasible_rate < threshold:
        recommendation = "feasible candidates scarce: test a smaller factor"
    else:
        recommendation = (
            f"class supply is adequate: keep current factor "
            f"{SAMPLING_EXPANSION_FACTOR:g}")
    report = {
        "scenario": int(ci),
        "sampling_expansion_factor": SAMPLING_EXPANSION_FACTOR,
        "target_total": int(CONFIG["initial_points"]),
        "forced_class_balance": True,
        "accepted_feasible": int(accepted_feasible),
        "accepted_infeasible": int(accepted_infeasible),
        "attempts": int(attempts),
        "classified_candidates": int(solved),
        "candidate_feasible": int(candidate_feasible),
        "candidate_infeasible": int(candidate_infeasible),
        "candidate_feasible_rate": float(feasible_rate),
        "candidate_infeasible_rate": float(1.0-feasible_rate),
        "solver_failures": int(solver_failures),
        "class_scarcity_threshold": threshold,
        "recommendation": recommendation,
        "automatic_factor_change": False,
    }
    report_path = data_path.with_name("sampling_diagnostic.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[scenario {ci:03d}/data] feasible={feasible_rate:.2%}, "
          f"accepted={accepted_feasible}/{accepted_infeasible}, "
          f"attempts={attempts}, solver_failures={solver_failures}; "
          f"{recommendation}")


def load_initial_data(path):
    with np.load(path) as data:
        return (np.asarray(data["points"], dtype=np.float32),
                np.asarray(data["labels"], dtype=np.float32))


def load_initial_generation_seconds(path):
    with np.load(path, allow_pickle=False) as data:
        if "generation_seconds" not in data.files:
            raise RuntimeError(
                f"The initial tag cache is missing the cumulative generation time, please use the timed version to regenerate it.: {path}")
        value = float(data["generation_seconds"])
    if not np.isfinite(value) or value < 0.0:
        raise RuntimeError(f"The cumulative generation time of the initial label is invalid: {value}")
    return value


def initialize_polytope(points, labels):
    A = fixed_normals()
    feasible = points[labels < 0]
    if not len(feasible):
        raise RuntimeError("There are no feasible points in the initial training set.")
    b = np.max(feasible @ A.T, axis=0) + 1e-3
    raw_scores = np.max(points @ A.T - b, axis=1)
    separated = raw_scores[(labels > 0) & (raw_scores > 0)]
    if len(separated):
        characteristic_distance = float(np.median(separated))
    else:
        characteristic_distance = float(np.median(np.abs(raw_scores)))
    initial_scale = 1.0 / max(characteristic_distance, 1e-3)
    return PaperPolytope(A, b, initial_scale).to(DEVICE)


def verify_one_update_point(oracle, point, A, b, expected_label,
                            expected_prediction):
    """Confirm that the candidate is indeed a target misclassification and distinguish physical labels from polygon predictions."""
    raw_violation = oracle.evaluate(point)
    if raw_violation is None or not np.isfinite(raw_violation):
        return (False, np.nan, np.nan, 0, np.nan,
                "physical_label_solver_failure")
    # The total relaxation is theoretically nonnegative; IPOPTMay give tiny negative numbers near the lower bound of a variable.
    violation = max(0.0, float(raw_violation))
    actual_label = -1 if violation <= FEASIBILITY_SLACK_TOL else 1
    prediction_score = float(np.max(np.asarray(A) @ point - np.asarray(b)))
    predicted_label = (-1 if prediction_score
                       <= POLYTOPE_CLASSIFICATION_TOL else 1)
    if actual_label != expected_label:
        reason = "physical_label_mismatch"
    elif predicted_label != expected_prediction:
        reason = "predicted_class_mismatch"
    else:
        reason = "accepted"
    return (reason == "accepted", violation, float(raw_violation),
            int(actual_label), prediction_score, reason)


def paper_update_points(A, b, over_base, over_floor, under_solver, oracle,
                        existing_points=None):
    """Only physical confirmations are returnedIF2F/F2IFpoint; route convergence is allowed when there is no corresponding error."""
    requested = CONFIG["update_points_per_route"]
    existing_points = np.asarray(
        [] if existing_points is None else existing_points,
        dtype=float).reshape(-1, 2)

    def nearest_distance(point, *accepted_groups):
        groups = [existing_points]
        groups.extend(np.asarray(group, dtype=float).reshape(-1, 2)
                      for group in accepted_groups if len(group))
        nonempty = [group for group in groups if len(group)]
        if not nonempty:
            return np.inf
        seen = np.concatenate(nonempty, axis=0)
        return float(np.min(np.linalg.norm(seen-point, axis=1)))

    # Take the original precision firstKKT+Slack continuation. If failed, enumerate the current two-dimensional
    # Predict the vertices of the polygon, initialized with the largest relaxation among the inner minimum relaxation solutions.
    # isomorphismKKTphysical variables. This is just a warm start, you still have to return eventuallytau=0of precisionKKTsolution.
    equation17_solver_attempts = []
    alpha1_result = over_base.solve(A, b)
    standard_metadata = dict(over_base.last_solve_metadata)
    standard_metadata["initialization"] = "standard"
    equation17_solver_attempts.append(standard_metadata)

    if alpha1_result is None:
        vertices = halfspace_polygon_vertices(A, b)
        best_vertex = None
        best_violation = -np.inf
        best_values = None
        successful_vertex_solves = 0
        for vertex in vertices:
            vertex_violation = oracle.evaluate(vertex)
            if vertex_violation is None or not np.isfinite(vertex_violation):
                continue
            successful_vertex_solves += 1
            if vertex_violation > best_violation:
                best_vertex = np.asarray(vertex, dtype=float).copy()
                best_violation = float(vertex_violation)
                best_values = oracle.kkt_warm_start_data()
        if best_values is not None:
            over_base.set_physical_warm_start(best_values)
            print("[KKT/equation17] Standard initializer failed; use predicted polygons"
                  f"Apex hot start ({successful_vertex_solves}/{len(vertices)}"
                  f"The inner layers are solved successfully, and the maximum relaxation={best_violation:.3e}) .")
            alpha1_result = over_base.solve(A, b)
            warm_metadata = dict(over_base.last_solve_metadata)
            warm_metadata.update({
                "initialization": "physical_vertex_warm_start",
                "vertex_count": int(len(vertices)),
                "successful_vertex_solves": int(successful_vertex_solves),
                "warm_start_vertex": best_vertex.tolist(),
                "warm_start_physical_slack": float(best_violation),
            })
            equation17_solver_attempts.append(warm_metadata)

    # Vertex physics warm start still fails when finally doing a small amount of precisionKKTcold start.
    for restart_index in range(1, MAX_EQUATION17_EXACT_RESTARTS + 1):
        if alpha1_result is not None:
            break
        print(f"[KKT/equation17] Try the{restart_index}/"
              f"{MAX_EQUATION17_EXACT_RESTARTS}an accurateKKTnew initial value.")
        alpha1_result = over_base.solve(
            A, b, restart_seed=SEED + 104729 * restart_index,
            allow_fallback=False)
        restart_metadata = dict(over_base.last_solve_metadata)
        restart_metadata["initialization"] = "jittered_exact_restart"
        restart_metadata["restart_index"] = int(restart_index)
        equation17_solver_attempts.append(restart_metadata)
    if alpha1_result is None:
        raise RuntimeError(
            "Essay style(17) KKTProblem solving failed: standard continuation, "
            "Predict polygon vertex physics hot start and"
            f"{MAX_EQUATION17_EXACT_RESTARTS}None of the accurate multiple initial values were successful..")
    kkt_solver_attempts = equation17_solver_attempts
    worst_over_point, alpha1, complementarity = alpha1_result
    if complementarity > KKT_COMPLEMENTARITY_TOL:
        raise RuntimeError(
            f"formula(17) KKTComplementary residuals are too large: {complementarity:.3e}")
    beta1 = BETA_FRACTION * max(alpha1, FEASIBILITY_SLACK_TOL)
    over, over_phi, over_rejections = [], [], []
    over_raw_phi, over_prediction_scores = [], []
    over_duplicate_flags, over_nearest_distances, over_sources = [], [], []
    over_label_failures = over_label_mismatches = 0
    over_prediction_mismatches = 0
    over_candidate_attempts = 1
    over_status = "active"
    over_retry_same_level = 0
    duplicate_distance = nearest_distance(worst_over_point)
    (accepted, physical_value, physical_raw, actual_label,
     prediction_score, reason) = verify_one_update_point(
        oracle, worst_over_point, A, b,
        expected_label=1, expected_prediction=-1)
    alpha1_physical = physical_value
    if accepted:
        over.append(worst_over_point)
        over_phi.append(physical_value)
        over_raw_phi.append(physical_raw)
        over_prediction_scores.append(prediction_score)
        over_duplicate_flags.append(
            bool(duplicate_distance <= UPDATE_DUPLICATE_POINT_TOL))
        over_nearest_distances.append(
            None if not np.isfinite(duplicate_distance)
            else float(duplicate_distance))
        over_sources.append("equation_17")
    else:
        over_label_failures += reason == "physical_label_solver_failure"
        over_label_mismatches += reason == "physical_label_mismatch"
        over_prediction_mismatches += reason == "predicted_class_mismatch"
        over_rejections.append({
            "attempt": 1, "source": "equation_17", "reason": reason,
            "point": np.asarray(worst_over_point, dtype=float).tolist(),
            "kkt_slack": float(alpha1),
            "physical_slack": float(physical_value),
            "physical_slack_raw": float(physical_raw),
            "actual_label": int(actual_label),
            "prediction_score": float(prediction_score),
            "is_duplicate": bool(
                duplicate_distance <= UPDATE_DUPLICATE_POINT_TOL),
            "nearest_point_distance": (
                None if not np.isfinite(duplicate_distance)
                else float(duplicate_distance)),
        })
        print(f"[update/IF2F] formula(17)Candidate failed physical review: {reason}, "
              f"physical_slack={physical_value:.3e}; Continue pressing(23)Supplementary harvesting.")
        if (reason == "physical_label_mismatch"
                and np.isfinite(physical_value)
                and physical_value <= FEASIBILITY_SLACK_TOL):
            # formula(17)is the worst-point estimate of the route; when independent review is still feasible, no more
            # artificially madeIF2Ftag, recording that the route has reached physical resolution.
            over_status = "converged_estimated_error_below_tolerance"
    previous = alpha1
    kkt_complementarity = [complementarity]
    while (over_status == "active" and len(over) < requested
           and over_candidate_attempts < MAX_UPDATE_CANDIDATE_ATTEMPTS):
        requested_floor = max(previous - beta1, 0.0)
        restart_seed = (None if over_retry_same_level == 0 else
                        SEED + 1009*over_candidate_attempts
                        + 9176*len(over))
        result = over_floor.solve(
            A, b, floor=requested_floor,
            restart_seed=restart_seed, commit=False)
        over_candidate_attempts += 1
        if result is None:
            kkt_solver_attempts.append(dict(over_floor.last_solve_metadata))
            over_rejections.append({
                "attempt": over_candidate_attempts,
                "source": "equation_23",
                "reason": "kkt_solver_failure",
                "requested_floor": float(requested_floor),
            })
            # Retry the same level with a different initial value until a valid point is found.
            over_retry_same_level += 1
            print(f"[update/IF2F] No.{over_candidate_attempts}candidatesKKTfailed, "
                  "Keep the current recursion level and change the initial value and try again.")
            continue
        kkt_solver_attempts.append(dict(over_floor.last_solve_metadata))
        point, value, comp = result
        if comp > KKT_COMPLEMENTARITY_TOL:
            raise RuntimeError(
                f"formula(23) KKTComplementary residuals are too large: {comp:.3e}")
        kkt_complementarity.append(comp)
        point = np.asarray(point, dtype=float)
        duplicate_distance = nearest_distance(point, over)
        (accepted, physical_value, physical_raw, actual_label,
         prediction_score, reason) = verify_one_update_point(
            oracle, point, A, b,
            expected_label=1, expected_prediction=-1)
        if accepted:
            over_floor.commit_current_solution(A, b, requested_floor)
            over.append(point)
            over_phi.append(physical_value)
            over_raw_phi.append(physical_raw)
            over_prediction_scores.append(prediction_score)
            over_duplicate_flags.append(
                bool(duplicate_distance <= UPDATE_DUPLICATE_POINT_TOL))
            over_nearest_distances.append(
                None if not np.isfinite(duplicate_distance)
                else float(duplicate_distance))
            over_sources.append("equation_23")
            # Recursively use the optimization value of the last generated point by paper. Duplicate points are allowed to count,
            # Equivalent to the formula(15)Increase the sample weight of the worst position.
            previous = value
            over_retry_same_level = 0
        else:
            # Do not submit the rejected point as a warm start, and do not reduce the recursion level; otherwise, it will be along the
            # same pseudo-borderKKTThe solution keeps slipping towards zero relaxation.
            over_retry_same_level += 1
            over_label_failures += reason == "physical_label_solver_failure"
            over_label_mismatches += reason == "physical_label_mismatch"
            over_prediction_mismatches += reason == "predicted_class_mismatch"
            over_rejections.append({
                "attempt": over_candidate_attempts,
                "source": "equation_23", "reason": reason,
                "point": np.asarray(point, dtype=float).tolist(),
                "kkt_slack": float(value),
                "physical_slack": float(physical_value),
                "physical_slack_raw": float(physical_raw),
                "actual_label": int(actual_label),
                "prediction_score": float(prediction_score),
                "requested_floor": float(requested_floor),
                "is_duplicate": bool(
                    duplicate_distance <= UPDATE_DUPLICATE_POINT_TOL),
                "nearest_point_distance": (
                    None if not np.isfinite(duplicate_distance)
                    else float(duplicate_distance)),
            })
            print(f"[update/IF2F] No.{over_candidate_attempts}candidates failed"
                  f"physical review: {reason}, physical_slack="
                  f"{physical_value:.3e}; Continue to collect more.")

    if over_status == "active" and len(over) < requested:
        equation23_rejections = [
            row for row in over_rejections
            if row.get("source") == "equation_23"
        ]
        physical_boundary_rejections = [
            row for row in equation23_rejections
            if (row.get("reason") == "physical_label_mismatch"
                and np.isfinite(row.get("physical_slack", np.nan))
                and float(row["physical_slack"])
                <= FEASIBILITY_SLACK_TOL)
        ]
        sequence_reached_physical_boundary = (
            len(over) > 0
            and over_candidate_attempts >= MAX_UPDATE_CANDIDATE_ATTEMPTS
            and len(equation23_rejections) > 0
            and len(physical_boundary_rejections)
            == len(equation23_rejections)
        )
        if sequence_reached_physical_boundary:
            over_status = "sequence_reached_physical_boundary"
            print(
                "[update/IF2F] formula(17)reserved"
                f"{len(over)}a realityIF2Fpoint; formula(23)in"
                f"{len(equation23_rejections)}All subsequent candidates are returned"
                "Feasible regions of independent physical models. This round of recursion terminates according to the physical boundary, "
                "No wrong labels will be added, and no supplements will be forced.10point.")
        else:
            raise RuntimeError(
                f"formula(17)Still detected realIF2F (physical alpha1="
                f"{alpha1_physical:.3e}) , Dan type(23)only get"
                f"{len(over)}/{requested}valid misclassification points; candidate attempts="
                f"{over_candidate_attempts}.Not all rejection reasons are returned"
                "The real feasible region, so it is still determined that the update solution failed..")
    if len(over) == requested:
        over_status = "requested_route_count_reached"

    hyperplane_results = []
    for i in range(N_SIDES):
        result = under_solver.solve(A[i], b[i])
        if result is not None:
            hyperplane_results.append((result[1], i, result[0]))
    if not hyperplane_results:
        raise RuntimeError("Essay style(21)All hyperplanes failed to solve.")
    alpha2, active_index, worst_under_point = max(
        hyperplane_results, key=lambda row: row[0])
    beta2 = BETA_FRACTION * max(alpha2, FEASIBILITY_SLACK_TOL)
    under, under_phi, under_rejections = [], [], []
    under_raw_phi, under_prediction_scores = [], []
    under_duplicate_flags, under_nearest_distances, under_sources = [], [], []
    under_label_failures = under_label_mismatches = 0
    under_prediction_mismatches = 0
    under_candidate_attempts = 1
    under_status = ("converged_estimated_error_below_tolerance"
                    if alpha2 <= POLYTOPE_CLASSIFICATION_TOL else "active")
    under_retry_same_level = 0
    if under_status == "active":
        duplicate_distance = nearest_distance(worst_under_point, over)
        (accepted, physical_value, physical_raw, actual_label,
         prediction_score, reason) = verify_one_update_point(
            oracle, worst_under_point, A, b,
            expected_label=-1, expected_prediction=1)
        if accepted:
            under.append(worst_under_point)
            under_phi.append(physical_value)
            under_raw_phi.append(physical_raw)
            under_prediction_scores.append(prediction_score)
            under_duplicate_flags.append(
                bool(duplicate_distance <= UPDATE_DUPLICATE_POINT_TOL))
            under_nearest_distances.append(
                None if not np.isfinite(duplicate_distance)
                else float(duplicate_distance))
            under_sources.append("equation_21")
        else:
            under_label_failures += reason == "physical_label_solver_failure"
            under_label_mismatches += reason == "physical_label_mismatch"
            under_prediction_mismatches += reason == "predicted_class_mismatch"
            under_rejections.append({
                "attempt": 1, "source": "equation_21", "reason": reason,
                "point": np.asarray(worst_under_point, dtype=float).tolist(),
                "polytope_violation": float(alpha2),
                "physical_slack": float(physical_value),
                "physical_slack_raw": float(physical_raw),
                "actual_label": int(actual_label),
                "prediction_score": float(prediction_score),
                "is_duplicate": bool(
                    duplicate_distance <= UPDATE_DUPLICATE_POINT_TOL),
                "nearest_point_distance": (
                    None if not np.isfinite(duplicate_distance)
                    else float(duplicate_distance)),
            })
            print(f"[update/F2IF] formula(21)Candidate failed physical review: {reason}, "
                  f"physical_slack={physical_value:.3e}; Keep the current level and try again.")
    previous = alpha2
    while (under_status == "active" and len(under) < requested
           and under_candidate_attempts < MAX_UPDATE_CANDIDATE_ATTEMPTS):
        restart_seed = (None if under_retry_same_level == 0 else
                        SEED + 2027*under_candidate_attempts
                        + 7919*len(under))
        result = under_solver.solve(
            A[active_index], b[active_index], cap=previous - beta2,
            restart_seed=restart_seed)
        under_candidate_attempts += 1
        if result is None:
            under_rejections.append({
                "attempt": under_candidate_attempts,
                "source": "equations_25_27",
                "reason": "true_domain_solver_failure",
                "requested_cap": float(previous - beta2),
            })
            under_retry_same_level += 1
            print(f"[update/F2IF] No.{under_candidate_attempts}candidate solutions failed, "
                  "Keep the current recursion level and change the initial value and try again.")
            continue
        point, value = result
        point = np.asarray(point, dtype=float)
        duplicate_distance = nearest_distance(point, over, under)
        (accepted, physical_value, physical_raw, actual_label,
         prediction_score, reason) = verify_one_update_point(
            oracle, point, A, b,
            expected_label=-1, expected_prediction=1)
        if accepted:
            under.append(point)
            under_phi.append(physical_value)
            under_raw_phi.append(physical_raw)
            under_prediction_scores.append(prediction_score)
            under_duplicate_flags.append(
                bool(duplicate_distance <= UPDATE_DUPLICATE_POINT_TOL))
            under_nearest_distances.append(
                None if not np.isfinite(duplicate_distance)
                else float(duplicate_distance))
            under_sources.append("equations_25_27")
            previous = value
            under_retry_same_level = 0
        else:
            under_retry_same_level += 1
            under_label_failures += reason == "physical_label_solver_failure"
            under_label_mismatches += reason == "physical_label_mismatch"
            under_prediction_mismatches += reason == "predicted_class_mismatch"
            under_rejections.append({
                "attempt": under_candidate_attempts,
                "source": "equations_25_27", "reason": reason,
                "point": np.asarray(point, dtype=float).tolist(),
                "polytope_violation": float(value),
                "physical_slack": float(physical_value),
                "physical_slack_raw": float(physical_raw),
                "actual_label": int(actual_label),
                "prediction_score": float(prediction_score),
                "is_duplicate": bool(
                    duplicate_distance <= UPDATE_DUPLICATE_POINT_TOL),
                "nearest_point_distance": (
                    None if not np.isfinite(duplicate_distance)
                    else float(duplicate_distance)),
            })
            print(f"[update/F2IF] No.{under_candidate_attempts}candidates failed"
                  f"physical review: {reason}, physical_slack="
                  f"{physical_value:.3e}; Continue to collect more.")

    if under_status == "active" and len(under) < requested:
        raise RuntimeError(
            f"formula(21)Still detected realF2IF (alpha2={alpha2:.3e}) , But"
            f"formula(25)-(27)only get{len(under)}/{requested}effective misclassification points; "
            f"candidate attempts={under_candidate_attempts}.This is an update solve failure.")
    if len(under) == requested:
        under_status = "requested_route_count_reached"

    over = np.asarray(over, dtype=float).reshape(-1, 2)
    over_phi = np.asarray(over_phi, dtype=float)
    under = np.asarray(under, dtype=float).reshape(-1, 2)
    under_phi = np.asarray(under_phi, dtype=float)
    metadata = {
        "alpha1": float(alpha1), "alpha2": float(alpha2),
        "alpha1_physical_at_estimated_worst": float(alpha1_physical),
        "beta1": float(beta1), "beta2": float(beta2),
        "active_hyperplane": int(active_index),
        "kkt_complementarity_max": float(max(kkt_complementarity)),
        "kkt_complementarity_passed": bool(
            max(kkt_complementarity) <= KKT_COMPLEMENTARITY_TOL),
        "kkt_solver_attempts": kkt_solver_attempts,
        "over_label_failures": int(over_label_failures),
        "under_label_failures": int(under_label_failures),
        "over_label_mismatches": int(over_label_mismatches),
        "under_label_mismatches": int(under_label_mismatches),
        "over_prediction_mismatches": int(over_prediction_mismatches),
        "under_prediction_mismatches": int(under_prediction_mismatches),
        "over_candidate_attempts": int(over_candidate_attempts),
        "under_candidate_attempts": int(under_candidate_attempts),
        "requested_points_per_route": int(requested),
        "accepted_over_points": int(len(over)),
        "accepted_under_points": int(len(under)),
        "over_duplicate_points": int(sum(over_duplicate_flags)),
        "under_duplicate_points": int(sum(under_duplicate_flags)),
        "over_is_duplicate": over_duplicate_flags,
        "under_is_duplicate": under_duplicate_flags,
        "over_nearest_point_distance": over_nearest_distances,
        "under_nearest_point_distance": under_nearest_distances,
        "over_point_sources": over_sources,
        "under_point_sources": under_sources,
        "over_physical_slack_raw": over_raw_phi,
        "under_physical_slack_raw": under_raw_phi,
        "over_prediction_scores": over_prediction_scores,
        "under_prediction_scores": under_prediction_scores,
        "over_sequence_indices": list(range(len(over))),
        "under_sequence_indices": list(range(len(under))),
        "over_expected_label": 1,
        "under_expected_label": -1,
        "all_accepted_points_physically_confirmed": True,
        "duplicates_retained_in_training": True,
        "duplicate_reference": (
            "all earlier training points plus points already accepted in the "
            "current round, using Euclidean distance in per-unit P-Q space"),
        "overcoverage_status": over_status,
        "undercoverage_status": under_status,
        "physical_sequence_stop_enabled": True,
        "physical_sequence_stop_rule": (
            "after the full candidate budget, retain at least one physically "
            "confirmed IF2F point and allow a short route only when every "
            "equation-(23) rejection is independently AC-feasible"),
        "max_candidate_attempts": int(MAX_UPDATE_CANDIDATE_ATTEMPTS),
        "over_rejections": over_rejections,
        "under_rejections": under_rejections,
    }
    points = np.concatenate((over, under), axis=0).astype(np.float32)
    labels = np.concatenate(
        (np.ones(len(over)), -np.ones(len(under)))).astype(np.float32)
    violations = np.concatenate((over_phi, under_phi))
    return points, labels, violations, metadata


def save_training_state(path, dtheta, reference, lower, upper, model,
                        points, labels, initial_history, update_histories,
                        update_metadata, initial_seconds, update_seconds,
                        initial_label_generation_seconds,
                        reference_generation_seconds, initial_context_seconds,
                        initialization_seconds,
                        initial_offline_seconds, update_offline_seconds,
                        completed_rounds, complete=False, A_initial=None,
                        b_initial=None, update_point_rows=None,
                        update_label_rows=None, update_violation_rows=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "dtheta": dtheta, "reference": reference,
        "sampling_lower": lower, "sampling_upper": upper,
        "A_raw": model.A.detach().cpu().numpy(),
        "b_raw": model.b.detach().cpu().numpy(),
        "log_scales_raw": model.log_scales.detach().cpu().numpy(),
        "training_points": points, "training_labels": labels,
        "initial_history": initial_history,
        "initial_training_seconds": np.asarray(initial_seconds),
        "update_training_seconds": np.asarray(update_seconds),
        "initial_label_generation_seconds": np.asarray(
            initial_label_generation_seconds),
        "reference_generation_seconds": np.asarray(
            reference_generation_seconds),
        "initial_context_seconds": np.asarray(initial_context_seconds),
        "initialization_seconds": np.asarray(initialization_seconds),
        "initial_offline_seconds": np.asarray(initial_offline_seconds),
        "update_offline_seconds": np.asarray(update_offline_seconds),
        "completed_rounds": np.asarray(completed_rounds),
        "update_metadata_json": np.asarray(json.dumps(update_metadata)),
        "complete": np.asarray(bool(complete)),
        "method_version": np.asarray(METHOD_VERSION),
        "training_config_json": np.asarray(json.dumps(
            training_cache_config(), sort_keys=True)),
    }
    for index, history in enumerate(update_histories, start=1):
        arrays[f"update_history_{index}"] = history
    optional_arrays = {
        "A_initial": A_initial,
        "b_initial": b_initial,
    }
    arrays.update({key: np.asarray(value) for key, value in
                   optional_arrays.items() if value is not None})
    if update_point_rows is not None:
        counts = np.asarray([len(row) for row in update_point_rows], dtype=int)
        largest_count = int(counts.max()) if counts.size else 0
        width = max(2 * CONFIG["update_points_per_route"], largest_count)
        packed_points = np.full(
            (len(update_point_rows), width, 2), np.nan, dtype=np.float32)
        packed_labels = np.zeros(
            (len(update_point_rows), width), dtype=np.float32)
        packed_violations = np.full(
            (len(update_point_rows), width), np.nan, dtype=float)
        for index, count in enumerate(counts):
            if count:
                packed_points[index, :count] = update_point_rows[index]
                packed_labels[index, :count] = update_label_rows[index]
                packed_violations[index, :count] = update_violation_rows[index]
        arrays.update({
            "update_points": packed_points,
            "update_labels": packed_labels,
            "update_physical_slack_violation": packed_violations,
            "update_round_counts": counts,
        })
        # Write a complete temporary file before atomically replacing the destination.
    # If the process is paused or terminated, existing breakpoints will not be zero-filled./incomplete ZIP File.
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def unpack_saved_update_rows(saved, completed_rounds=None):
    """Read the update points that actually pass physical review in each round in the fixed-length padding format."""
    stored_points = np.asarray(saved["update_points"], dtype=np.float32)
    stored_labels = np.asarray(saved["update_labels"], dtype=np.float32)
    stored_violations = np.asarray(
        saved["update_physical_slack_violation"], dtype=float)
    rounds = len(stored_points) if completed_rounds is None else completed_rounds
    saved_files = set(saved.files) if hasattr(saved, "files") else set(saved)
    if "update_round_counts" in saved_files:
        counts = np.asarray(saved["update_round_counts"], dtype=int)[:rounds]
    else:
        counts = np.full(rounds, stored_points.shape[1], dtype=int)
    point_rows = [stored_points[index, :count].copy()
                  for index, count in enumerate(counts)]
    label_rows = [stored_labels[index, :count].copy()
                  for index, count in enumerate(counts)]
    violation_rows = [stored_violations[index, :count].copy()
                      for index, count in enumerate(counts)]
    return point_rows, label_rows, violation_rows


def train_condition(ci, case, ppc, dtheta, reference, overwrite=False):
    result_path = MODEL_DIR / f"scenario_{ci:03d}" / "result.npz"
    if result_path.exists() and not overwrite:
        with np.load(result_path, allow_pickle=False) as old:
            if cache_matches_training_config(old, dtheta, reference):
                print(f"[scenario {ci:03d}] Use full cache.")
                return {key: old[key] for key in old.files}
            print(f"[scenario {ci:03d}] The cache configuration is inconsistent with the current experiment, "
                  "Automatically regenerate with current settings.")

    reference_started = time.perf_counter()
    reference_calculator = shared.ReferencePointCalculator(
        ppc, case["params"]["params_dict"],
        original_model=case["errorcalculator"].original_model)
    measured_reference = reference_calculator.compute(dtheta)
    reference_generation_seconds = time.perf_counter() - reference_started
    if (measured_reference is None
            or not np.allclose(measured_reference, reference,
                               rtol=0.0, atol=1e-7)):
        raise RuntimeError(
            f"Working conditions{ci:03d}Recalculated training reference points are inconsistent with public case cache.")

    context_started = time.perf_counter()
    ec, lower, upper, radii = prepare_condition_context(
        case, ppc, dtheta, reference)
    oracle = FeasibilitySlackOracle(ec.original_model)
    context_seconds = time.perf_counter() - context_started
    initial_path = generate_initial_data(
        ci, oracle, lower, upper, overwrite=overwrite)
    initial_generation_seconds = load_initial_generation_seconds(initial_path)
    initial_points, initial_labels = load_initial_data(initial_path)
    initialization_started = time.perf_counter()
    model = initialize_polytope(initial_points, initial_labels)
    initialization_seconds = time.perf_counter() - initialization_started
    initial_history, initial_seconds = fit_learning_module(
        model, initial_points, initial_labels, f"scenario{ci:03d}/initial")
    A_initial, b_initial = normalize_halfspaces(
        model.A.detach().cpu().numpy(), model.b.detach().cpu().numpy())
    initial_offline_seconds = (
        reference_generation_seconds + context_seconds
        + initial_generation_seconds
        + initialization_seconds + initial_seconds)

    # The paper update optimization model is only constructed once for each fixed operating condition., 50wheel update onlyA, bparameters.
    update_stage_started = time.perf_counter()
    over_base = KKTOvercoverageUpdater(ec.original_model, False)
    over_floor = KKTOvercoverageUpdater(ec.original_model, True)
    under_solver = TrueDomainHyperplaneUpdater(ec.original_model)
    points, labels = initial_points.copy(), initial_labels.copy()
    update_histories, update_metadata = [], []
    update_seconds = 0.0
    previous_update_offline_seconds = 0.0

    def cumulative_update_offline_seconds():
        return (previous_update_offline_seconds
                + time.perf_counter() - update_stage_started)
    update_point_rows, update_label_rows, update_violation_rows = [], [], []
    state_path = result_path.with_name("training_state.npz")
    first_round = 1
    state_is_readable = state_path.exists()
    if state_is_readable and not overwrite:
        try:
            with np.load(state_path, allow_pickle=False) as state_probe:
                # actually read a key field instead of just checking ZIP Directory to identify the interrupted
                # Zero padding left behind or partial writing to the file.
                int(state_probe["completed_rounds"])
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
            state_is_readable = False
            print(f"[scenario {ci:03d}] training_state.npz damaged, "
                  f"Ignore this breakpoint and rebuild from the beginning of this case: {exc}")
    if state_is_readable and not overwrite:
        with np.load(state_path, allow_pickle=False) as state:
            resumable = (
                cache_matches_training_config(
                    state, dtheta, reference, require_complete=False)
                and 0 < int(state["completed_rounds"])
                    < CONFIG["update_rounds"]
                and np.allclose(state["sampling_lower"], lower,
                                rtol=0.0, atol=1e-12)
                and np.allclose(state["sampling_upper"], upper,
                                rtol=0.0, atol=1e-12)
            )
            if resumable:
                completed = int(state["completed_rounds"])
                with torch.no_grad():
                    model.A.copy_(torch.as_tensor(
                        state["A_raw"], dtype=torch.float32, device=DEVICE))
                    model.b.copy_(torch.as_tensor(
                        state["b_raw"], dtype=torch.float32, device=DEVICE))
                    model.log_scales.copy_(torch.as_tensor(
                        state["log_scales_raw"], dtype=torch.float32,
                        device=DEVICE))
                points = np.asarray(state["training_points"], dtype=np.float32)
                labels = np.asarray(state["training_labels"], dtype=np.float32)
                initial_history = np.asarray(state["initial_history"], dtype=float)
                initial_seconds = float(state["initial_training_seconds"])
                update_seconds = float(state["update_training_seconds"])
                timing_fields = {
                    "initial_label_generation_seconds",
                    "reference_generation_seconds",
                    "initial_context_seconds", "initialization_seconds",
                    "initial_offline_seconds", "update_offline_seconds",
                }
                if not timing_fields.issubset(state.files):
                    raise RuntimeError(
                        f"Working conditions{ci:03d}The breakpoint is missing the cumulative offline timing field and cannot be continued..")
                initial_generation_seconds = float(
                    state["initial_label_generation_seconds"])
                reference_generation_seconds = float(
                    state["reference_generation_seconds"])
                context_seconds = float(state["initial_context_seconds"])
                initialization_seconds = float(
                    state["initialization_seconds"])
                initial_offline_seconds = float(
                    state["initial_offline_seconds"])
                previous_update_offline_seconds = float(
                    state["update_offline_seconds"])
                update_metadata = json.loads(str(state["update_metadata_json"]))
                update_histories = [np.asarray(
                    state[f"update_history_{index}"], dtype=float)
                    for index in range(1, completed + 1)]
                points_per_round = 2 * CONFIG["update_points_per_route"]
                if {"update_points", "update_labels",
                        "update_physical_slack_violation",
                        "update_round_counts"}.issubset(state.files):
                    stored_points = np.asarray(
                        state["update_points"], dtype=np.float32)
                    stored_labels = np.asarray(
                        state["update_labels"], dtype=np.float32)
                    stored_violations = np.asarray(
                        state["update_physical_slack_violation"], dtype=float)
                    counts = np.asarray(
                        state["update_round_counts"], dtype=int)[:completed]
                    if (len(counts) != completed
                            or np.any(counts < 0)
                            or np.any(counts > points_per_round)):
                        raise RuntimeError(
                            f"Working conditions{ci:03d}The number of breakpoint update points exceeds the valid range.")
                    update_point_rows = [
                        stored_points[index, :count].copy()
                        for index, count in enumerate(counts)]
                    update_label_rows = [
                        stored_labels[index, :count].copy()
                        for index, count in enumerate(counts)]
                    update_violation_rows = [
                        stored_violations[index, :count].copy()
                        for index, count in enumerate(counts)]
                else:
                    raise RuntimeError(
                        f"Working conditions{ci:03d}The breakpoint is missing the update protocol audit array and cannot be continued..")
                if len(update_metadata) != completed:
                    raise RuntimeError(
                        f"Working conditions{ci:03d}Breakpoint update metadata rounds are incomplete.")
                for saved_round, (row_labels, item) in enumerate(
                        zip(update_label_rows, update_metadata), start=1):
                    expected_count = (
                        int(item.get("accepted_over_points", -1))
                        + int(item.get("accepted_under_points", -1)))
                    if (len(row_labels) != expected_count
                            or np.any(np.asarray(row_labels) == 0)):
                        raise RuntimeError(
                            f"Working conditions{ci:03d}breakpoint{saved_round}wheel label with"
                            "Update metadata is inconsistent.")
                first_round = completed + 1
                print(f"[scenario {ci:03d}] fromtraining_state.npzContinue calculation: "
                      f"Completed{completed}/{CONFIG['update_rounds']}wheel, "
                      f"next round={first_round}.")

            # Reject same-version checkpoints whose values fall outside the configured angular sector.
        # Physically confirmed sample recovery geometry; this does not introduce update points from older methods.
        if (first_round > 1
                and not halfspaces_within_normal_sectors(
                    model.A.detach().cpu().numpy())):
            print(f"[scenario {ci:03d}] Degenerate normal in breakpoint detected, "
                  "Performs a bounded geometry recovery using existing labels (does not recalculate physical labels) .")
            project_model_halfspaces(model)
            repair_history, repair_seconds = fit_learning_module(
                model, points, labels,
                f"scenario{ci:03d}/bounded_geometry_repair")
            normalize_halfspaces(
                model.A.detach().cpu().numpy(),
                model.b.detach().cpu().numpy())
            update_seconds += repair_seconds
            if update_metadata:
                update_metadata[-1]["bounded_geometry_repair"] = True
                update_metadata[-1]["bounded_geometry_repair_steps"] = int(
                    repair_history[-1, 0]) if len(repair_history) else 0
            save_training_state(
                state_path, dtheta, reference, lower, upper, model,
                points, labels, initial_history, update_histories,
                update_metadata, initial_seconds, update_seconds,
                initial_generation_seconds, reference_generation_seconds,
                context_seconds,
                initialization_seconds,
                initial_offline_seconds,
                cumulative_update_offline_seconds(),
                first_round-1, complete=False,
                A_initial=A_initial, b_initial=b_initial,
                update_point_rows=update_point_rows,
                update_label_rows=update_label_rows,
                update_violation_rows=update_violation_rows)

    for round_index in range(first_round, CONFIG["update_rounds"] + 1):
        round_started = time.perf_counter()
        print(f"[scenario {ci:03d}] paper update {round_index}/"
              f"{CONFIG['update_rounds']}")
        A, b = normalize_halfspaces(
            model.A.detach().cpu().numpy(), model.b.detach().cpu().numpy())
        new_points, new_labels, new_violations, metadata = paper_update_points(
            A, b, over_base, over_floor, under_solver, oracle,
            existing_points=points)
        points = np.concatenate((points, new_points), axis=0)
        labels = np.concatenate((labels, new_labels), axis=0)
        history, seconds = fit_learning_module(
            model, points, labels, f"scenario{ci:03d}/update{round_index}")
        update_seconds += seconds
        update_histories.append(history)
        metadata["round"] = round_index
        metadata["round_offline_seconds"] = (
            time.perf_counter() - round_started)
        update_metadata.append(metadata)
        update_point_rows.append(new_points)
        update_label_rows.append(new_labels)
        update_violation_rows.append(new_violations)
        save_training_state(
            state_path, dtheta, reference, lower, upper, model, points, labels,
            initial_history, update_histories, update_metadata,
            initial_seconds, update_seconds, initial_generation_seconds,
            reference_generation_seconds, context_seconds,
            initialization_seconds, initial_offline_seconds,
            cumulative_update_offline_seconds(), round_index, complete=False,
            A_initial=A_initial, b_initial=b_initial,
            update_point_rows=update_point_rows,
            update_label_rows=update_label_rows,
            update_violation_rows=update_violation_rows)

    A_final, b_final = normalize_halfspaces(
        model.A.detach().cpu().numpy(), model.b.detach().cpu().numpy())
    save_training_state(
        result_path, dtheta, reference, lower, upper, model, points, labels,
        initial_history, update_histories, update_metadata,
        initial_seconds, update_seconds, initial_generation_seconds,
        reference_generation_seconds, context_seconds,
        initialization_seconds, initial_offline_seconds,
        cumulative_update_offline_seconds(), CONFIG["update_rounds"],
        complete=True,
        A_initial=A_initial, b_initial=b_initial,
        update_point_rows=update_point_rows,
        update_label_rows=update_label_rows,
        update_violation_rows=update_violation_rows)
    with np.load(result_path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays.update({
        "A_initial": A_initial, "b_initial": b_initial,
        "A_final": A_final, "b_final": b_final,
        "device_radii": radii,
    })
    np.savez(result_path, **arrays)
    # A completed result supersedes the resumable checkpoint.
    state_path.unlink(missing_ok=True)
    return arrays


def measure_exact_lookup_once(model, condition):
    """Measure the online polygon query time of a trained operating condition according to the original caliber.

    model/Polygon library is loaded early; timing includes numpy input transfer Tensor, Incoming calculation
    Equipment, an accurate operating condition query and complete A, b transfer back CPU numpy.No preheating,
    No repeated averaging is performed, and model loading time is not included. Output transfer back CPU will wait CUDA
    The operation is complete. This caliber is the same as R1.6 ``measure_online_time`` consistent.
    """
    model.eval()
    condition = np.asarray(condition, dtype=np.float64)
    with torch.no_grad():
        started = time.time()
        tensor = torch.tensor(condition, dtype=torch.float32).to(DEVICE)
        A, b = model(tensor)
        _ = A.detach().cpu().numpy(), b.detach().cpu().numpy()
        return time.time() - started


def save_summary(rows, config):
    OUT.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = OUT / "summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_completed_condition(ci, dtheta, reference):
    """Load one completed shard result without silently training missing work."""
    result_path = MODEL_DIR / f"scenario_{ci:03d}" / "result.npz"
    if not result_path.exists():
        raise FileNotFoundError(f"Working conditions{ci:03d}Not trained yet: {result_path}")
    with np.load(result_path, allow_pickle=True) as saved:
        if not cache_matches_training_config(saved, dtheta, reference):
            raise RuntimeError(
                f"Working conditions{ci:03d}If the result is incomplete or the configuration is inconsistent, please rerun its training shard..")
        required = {
            "A_initial", "b_initial", "A_final", "b_final",
            "initial_training_seconds", "update_training_seconds",
            "initial_label_generation_seconds", "reference_generation_seconds",
            "initial_context_seconds",
            "initialization_seconds",
            "initial_offline_seconds", "update_offline_seconds",
            "training_points", "training_labels", "update_points",
            "update_labels", "update_physical_slack_violation",
            "update_round_counts", "update_metadata_json",
        }
        missing = required.difference(saved.files)
        if missing:
            raise RuntimeError(
                f"Working conditions{ci:03d}Results are missing fields: {sorted(missing)}")
        timing = {key: float(saved[key]) for key in (
            "initial_training_seconds", "update_training_seconds",
            "initial_label_generation_seconds", "reference_generation_seconds",
            "initial_context_seconds",
            "initialization_seconds", "initial_offline_seconds",
            "update_offline_seconds")}
        if any(not np.isfinite(value) or value < 0.0
               for value in timing.values()):
            raise RuntimeError(f"Working conditions{ci:03d}Contains invalid cumulative time: {timing}")
        if (timing["initial_offline_seconds"] + 1e-9
                < timing["initial_training_seconds"]):
            raise RuntimeError(f"Working conditions{ci:03d}The initial offline time is less than the learning time.")
        if (timing["update_offline_seconds"] + 1e-9
                < timing["update_training_seconds"]):
            raise RuntimeError(f"Working conditions{ci:03d}Update offline time is less than learning time.")
        return {key: saved[key] for key in saved.files}


def get_shard_indices(total, shard_index, num_shards):
    if num_shards < 1:
        raise ValueError("num-shardsMust be at least1.")
    if shard_index is None:
        if num_shards != 1:
            raise ValueError("num-shardsgreater than1must be specified at the same timeshard-index.")
        return list(range(total))
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard-indexMust be located in[0, {num_shards - 1}].")
    return [int(value) for value in np.array_split(
        np.arange(total, dtype=int), num_shards)[shard_index]]


def main(stage="diagnostic", overwrite=False, shard_index=None,
         num_shards=1, train_only=False, finalize_only=False,
         prepare_only=False):
    configure_stage(stage)
    selected_modes = sum(bool(value) for value in (
        train_only, finalize_only, prepare_only))
    if selected_modes > 1:
        raise ValueError("prepare-only, train-onlyandfinalize-onlycannot be used at the same time.")
    print(f"R1.1The two-dimensional two-module reproduction of the paper | stage={STAGE} | device={DEVICE}")
    print(f"Results directory={OUT}")
    print("Offline timing=50Accumulation of operating conditions, including learning and physical data generation, and support for accumulation of breakpoints")
    print(f"Common operating conditions={CONFIG['n_conditions']}; initial point/Working conditions="
          f"{CONFIG['initial_points']}; Update round number={CONFIG['update_rounds']}")
    update_description = (
        f"Maximum per route per round{CONFIG['update_points_per_route']}a reality"
        "Misclassified points; the error is lower than the threshold or formula(23)Return all after complete recursion"
        "When the real feasible region is reached, the route is allowed to end with actual valid points.; "
        f"Maximum attempts per route candidate={MAX_UPDATE_CANDIDATE_ATTEMPTS}")
    print(f"Sampling box expansion factor={SAMPLING_EXPANSION_FACTOR:g}; "
          "Initial label is mandatory/Not feasible, half and half; "
          "learning loss=Essay style(15)The overall average of the whole samplehinge; "
          f"{update_description}")
    ppc, case = shared.build_case()
    conditions = load_common_conditions(case['params']['count'])
    selected_indices = get_shard_indices(
        len(conditions), shard_index, num_shards)

    if prepare_only:
        truth_path = load_or_prepare_common_truth(
            case, ppc, conditions, overwrite)
        print(f"The public true feasible region is ready: {truth_path}")
        return truth_path

    # Ten training shards will be started at the same time, and they must not be allowed to compete to write the same true feasible region file.
    # Therefore, it must be run separately before parallel training.preparescript.
    if (train_only and STAGE == "formal"
            and not formal_truth_cache_matches(conditions)):
        raise FileNotFoundError(
            "missing match current50The common true feasible region of operating conditions and evaluation directions. Please run it separately first"
            "main_revise1_1_learning_comparison_prepare.py, Start sharding in parallel again.\n"
            f"object file: {R16_TRUTH_PATH}")
    if train_only and STAGE == "formal":
        truth_path = R16_TRUTH_PATH
        print(f"[evaluate] Sharded read-only public true feasible region: {truth_path}")
    else:
        truth_path = load_or_prepare_common_truth(
            case, ppc, conditions, overwrite)
    with np.load(truth_path) as truth:
        if not np.allclose(truth["dtheta"], conditions, rtol=0.0, atol=1e-7):
            raise ValueError("R1.6The test true value and the test label are not the same set of operating conditions.")
        references = np.asarray(truth["x_ref"], dtype=float)
        reference_success = np.asarray(truth["reference_success"], dtype=bool)
    if not np.all(reference_success):
        failed = np.where(~reference_success)[0].tolist()
        raise RuntimeError(f"Common test condition reference point failed: {failed}")

    if finalize_only:
        results = [
            load_completed_condition(ci, conditions[ci], references[ci])
            for ci in range(len(conditions))
        ]
        print(f"Completely read{len(results)}operating conditions, start unified evaluation.")
    else:
        if shard_index is not None:
            print(f"training shards={shard_index + 1}/{num_shards}; "
                  f"Working condition index={selected_indices}")
        results = []
        for local_index, ci in enumerate(selected_indices, start=1):
            print("\n" + "="*78)
            print(f"Independent operating conditions {ci + 1}/{len(conditions)}; "
                  f"Progress of this shard {local_index}/{len(selected_indices)}")
            results.append(train_condition(
                ci, case, ppc, conditions[ci], references[ci],
                overwrite=overwrite))

        if train_only:
            manifest_dir = OUT / "shards"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / (
                f"shard_{shard_index + 1:02d}_of_{num_shards:02d}.json")
            manifest_path.write_text(json.dumps({
                "stage": STAGE,
                "shard_index_zero_based": shard_index,
                "num_shards": num_shards,
                "scenario_indices": selected_indices,
                "scenario_count": len(selected_indices),
                "complete": True,
                "training_config": training_cache_config(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nTraining sharding completed: {manifest_path}")
            print("This shard does not write polygon libraries, evaluation orsummary, Avoid concurrent coverage.")
            return manifest_path

        if len(results) != len(conditions):
            raise RuntimeError("Nottrain-onlyThe pattern must include all operating conditions and cannot only summarize a single shard..")

    A_initial = np.stack([np.asarray(row["A_initial"]) for row in results])
    b_initial = np.stack([np.asarray(row["b_initial"]) for row in results])
    A_final = np.stack([np.asarray(row["A_final"]) for row in results])
    b_final = np.stack([np.asarray(row["b_final"]) for row in results])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bank_path = MODEL_DIR / "independent_polygon_bank.npz"
    np.savez(bank_path, dtheta=conditions,
             A_initial=A_initial, b_initial=b_initial,
             A_updated=A_final, b_updated=b_final,
             theta_used_as_learning_input=np.asarray(False),
             exact_scenario_lookup=np.asarray(True),
             method_version=np.asarray(METHOD_VERSION))

    initial_bank = ExactScenarioPolytopeBank(
        conditions, A_initial, b_initial).to(DEVICE)
    final_bank = ExactScenarioPolytopeBank(
        conditions, A_final, b_final).to(DEVICE)
    # LinThe method is notthetaGeneralized networks can only query conditions that have been trained on a case-by-case basis. So online
    # Time a trained operating condition instead of the evaluator's default all-zero probe.
    initial_online = measure_exact_lookup_once(initial_bank, conditions[0])
    final_online = measure_exact_lookup_once(final_bank, conditions[0])
    initial_metrics = shared.evaluate_model(
        "lin_paper_initial", initial_bank, case, truth_path,
        None, evaluation_dir=EVAL_DIR,
        online_time_override=initial_online, theta_override=conditions)
    final_metrics = shared.evaluate_model(
        "lin_paper_updated", final_bank, case, truth_path,
        None, evaluation_dir=EVAL_DIR,
        online_time_override=final_online, theta_override=conditions)
    initial_train_seconds = sum(
        float(row["initial_training_seconds"]) for row in results)
    update_train_seconds = sum(
        float(row["update_training_seconds"]) for row in results)
    initial_offline_seconds = sum(
        float(row["initial_offline_seconds"]) for row in results)
    update_offline_seconds = sum(
        float(row["update_offline_seconds"]) for row in results)
    initial_label_generation_seconds = sum(
        float(row["initial_label_generation_seconds"]) for row in results)
    initial_context_seconds = sum(
        float(row["initial_context_seconds"]) for row in results)
    reference_generation_seconds = sum(
        float(row["reference_generation_seconds"]) for row in results)
    initialization_seconds = sum(
        float(row["initialization_seconds"]) for row in results)
    if initial_offline_seconds + 1e-9 < initial_train_seconds:
        raise RuntimeError("The initial accumulated offline time is less than the learning time included in it.")
    if update_offline_seconds + 1e-9 < update_train_seconds:
        raise RuntimeError("The cumulative offline time of the update is less than the learning time included in it.")
    total_initial_points = len(conditions) * CONFIG["initial_points"]
    total_initial_feasible = sum(int(np.count_nonzero(
        np.asarray(result["training_labels"])[
            :CONFIG["initial_points"]] < 0)) for result in results)
    total_initial_infeasible = total_initial_points - total_initial_feasible
    total_update_points = sum(
        max(0, len(np.asarray(result["training_labels"]))
            - CONFIG["initial_points"])
        for result in results)

    maximum_update_points = (
        len(conditions) * CONFIG["update_rounds"]
        * 2 * CONFIG["update_points_per_route"])
    update_metadata_all = []
    update_audit_records = []
    total_update_feasible = total_update_infeasible = 0
    for ci, result in enumerate(results):
        result_labels = np.asarray(result["training_labels"], dtype=float)
        update_labels = result_labels[CONFIG["initial_points"]:]
        total_update_feasible += int(np.count_nonzero(update_labels < 0))
        total_update_infeasible += int(np.count_nonzero(update_labels > 0))
        metadata = json.loads(str(result["update_metadata_json"]))
        if len(metadata) != CONFIG["update_rounds"]:
            raise RuntimeError(
                f"Working conditions{ci:03d}Update metadata round error: "
                f"{len(metadata)} != {CONFIG['update_rounds']}.")
        for round_index, item in enumerate(metadata):
            round_seconds = float(item.get("round_offline_seconds", np.nan))
            if not np.isfinite(round_seconds) or round_seconds < 0.0:
                raise RuntimeError(
                    f"Working conditions{ci:03d}No.{round_index + 1}Round missing valid offline timing.")
            for prefix in ("over", "under"):
                accepted = int(item.get(f"accepted_{prefix}_points", -1))
                status_key = ("overcoverage_status" if prefix == "over"
                              else "undercoverage_status")
                status = str(item.get(status_key, ""))
                valid = (
                    accepted == CONFIG["update_points_per_route"]
                    or (accepted < CONFIG["update_points_per_route"]
                        and status in {
                            "converged_estimated_error_below_tolerance",
                            "sequence_reached_physical_boundary",
                        })
                )
                if not valid:
                    raise RuntimeError(
                        f"Working conditions{ci:03d}No.{round_index + 1}wheel{prefix}route"
                        f"Inconsistent status: accepted={accepted}, status={status!r}.")
            update_audit_records.append({
                "scenario": int(ci),
                "round": int(round_index + 1),
                **item,
            })
        update_metadata_all.extend(metadata)
    metadata_update_points = sum(
        int(item.get("accepted_over_points", 0))
        + int(item.get("accepted_under_points", 0))
        for item in update_metadata_all)
    if total_update_points != metadata_update_points:
        raise RuntimeError(
            "Training array is inconsistent with updated metadata points: "
            f"{total_update_points} != {metadata_update_points}.")

    update_audit = {
        "update_points_maximum_if_both_routes_active": int(
            maximum_update_points),
        "update_points_actual": int(total_update_points),
        "update_feasible_points": int(total_update_feasible),
        "update_infeasible_points": int(total_update_infeasible),
        "over_duplicate_points": int(sum(
            int(item.get("over_duplicate_points", 0))
            for item in update_metadata_all)),
        "under_duplicate_points": int(sum(
            int(item.get("under_duplicate_points", 0))
            for item in update_metadata_all)),
        "over_candidate_attempts": int(sum(
            int(item.get("over_candidate_attempts", 0))
            for item in update_metadata_all)),
        "under_candidate_attempts": int(sum(
            int(item.get("under_candidate_attempts", 0))
            for item in update_metadata_all)),
        "over_label_mismatches": int(sum(
            int(item.get("over_label_mismatches", 0))
            for item in update_metadata_all)),
        "under_label_mismatches": int(sum(
            int(item.get("under_label_mismatches", 0))
            for item in update_metadata_all)),
        "over_prediction_mismatches": int(sum(
            int(item.get("over_prediction_mismatches", 0))
            for item in update_metadata_all)),
        "under_prediction_mismatches": int(sum(
            int(item.get("under_prediction_mismatches", 0))
            for item in update_metadata_all)),
        "over_label_solver_failures": int(sum(
            int(item.get("over_label_failures", 0))
            for item in update_metadata_all)),
        "under_label_solver_failures": int(sum(
            int(item.get("under_label_failures", 0))
            for item in update_metadata_all)),
        "over_converged_rounds": int(sum(
            item.get("overcoverage_status") ==
            "converged_estimated_error_below_tolerance"
            for item in update_metadata_all)),
        "over_physical_boundary_stopped_rounds": int(sum(
            item.get("overcoverage_status") ==
            "sequence_reached_physical_boundary"
            for item in update_metadata_all)),
        "under_converged_rounds": int(sum(
            item.get("undercoverage_status") ==
            "converged_estimated_error_below_tolerance"
            for item in update_metadata_all)),
        "update_round_processing_seconds": float(sum(
            float(item.get("round_offline_seconds", 0.0))
            for item in update_metadata_all)),
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    update_audit_path = EVAL_DIR / "update_protocol_summary.json"
    update_audit_path.write_text(json.dumps({
        "method_version": METHOD_VERSION,
        "summary": update_audit,
        "scenario_count": len(results),
        "round_record_count": len(update_audit_records),
        "detail_source": "models/scenario_*/result.npz:update_metadata_json",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    def row(method, metrics, learning_seconds, online_seconds, offline_seconds,
            point_count, include_updates):
        result = {"method": method, **metrics}
        result.update({
            "training_conditions": len(conditions),
            "test_conditions": len(conditions),
            "independent_polygons": len(conditions),
            "theta_used_as_learning_input": False,
            "initial_points_per_condition": CONFIG["initial_points"],
            "initial_class_balance_forced": True,
            "initial_feasible_points": total_initial_feasible,
            "initial_infeasible_points": total_initial_infeasible,
            "update_rounds": CONFIG["update_rounds"],
            "update_points_per_route": CONFIG["update_points_per_route"],
            "point_labels_generated": point_count,
            "point_labels_final_training": point_count,
            **(update_audit if include_updates else {
                key: 0 for key in update_audit}),
            "learning_seconds": learning_seconds,
            "initial_label_generation_seconds": (
                initial_label_generation_seconds),
            "reference_generation_seconds": reference_generation_seconds,
            "initial_context_seconds": initial_context_seconds,
            "initialization_seconds": initialization_seconds,
            "update_offline_seconds": (
                update_offline_seconds if include_updates else 0.0),
            "online_inference_seconds": online_seconds,
            "offline_total_seconds": offline_seconds,
            "offline_total_includes_learning": True,
            "offline_timing_checkpoint_safe": True,
            "symmetric_radial_error": (
                metrics["undercoverage_mean"] + metrics["overcoverage_mean"]),
        })
        return result
    rows = [
        row("lin_paper_initial", initial_metrics, initial_train_seconds,
            initial_online, initial_offline_seconds, total_initial_points,
            include_updates=False),
        row("lin_paper_updated", final_metrics,
            initial_train_seconds + update_train_seconds, final_online,
            initial_offline_seconds + update_offline_seconds,
            total_initial_points + total_update_points,
            include_updates=True),
    ]
    summary_path = save_summary(rows, {
        "case": CASENAME, "stage": STAGE,
        "method": "Lin et al. two-module method adapted only in output dimension to 2D TDI",
        "common_condition_path": str(COMMON_TEST_CONDITION_PATH),
        "common_r16_truth_path": str(truth_path),
        "conditions": len(conditions), "n_sides": N_SIDES,
        "initial_points_per_condition": CONFIG["initial_points"],
        "initial_sampling": (
            "draw iid uniform candidates and retain the first 5000 feasible "
            "and first 5000 infeasible successfully labelled points"),
        "initial_class_balance_forced": True,
        "feasibility_label": "minimum total operational-bound slack",
        "feasibility_slack_tolerance": FEASIBILITY_SLACK_TOL,
        "learning_loss": (
            "paper equation (15): one global arithmetic mean of hinge losses "
            "over all training points, with positive per-row score scales; "
            "no class reweighting, radial/model-selection loss, or L2"),
        "weight_selection": (
            f"fixed {CONFIG['max_learning_steps']} optimization steps per fit"),
        "overcoverage_update": "equations (17)/(23), inner KKT reformulation",
        "kkt_solver_protocol": (
            "solve exact complementarity first; on IPOPT failure use "
            "tau continuation 1e-3,1e-4,1e-5,1e-6,0; optionally continue "
            "from previous A,b; accept only tau=0 with complementarity "
            f"residual <= {KKT_COMPLEMENTARITY_TOL:g}"),
        "undercoverage_update": "equations (21)/(25)-(27)",
        "beta1_fraction": BETA_FRACTION,
        "beta2_fraction": BETA_FRACTION,
        "update_rounds": CONFIG["update_rounds"],
        "update_points_per_route": CONFIG["update_points_per_route"],
        "update_protocol": (
            "only physically verified misclassification points enter training; "
            "a rejected candidate retries the same recursive level from a "
            "perturbed start; a route may return fewer points when its estimated "
            "worst error is below tolerance, or when at least one true IF2F "
            "point was retained and every equation-(23) rejection over the full "
            "candidate budget is independently AC-feasible"),
        "duplicate_update_outputs": (
            "retained as separate training samples and fully audited by per-point "
            "duplicate flags and nearest-previous-point distances"),
        "update_route_interpretation": (
            "one base output from equation (17)/(21), followed by recursive "
            "outputs from equation (23)/(25)-(27); only true IF2F/F2IF points "
            "are labels, while route-mismatched candidates are diagnostics"),
        "update_audit": update_audit,
        "update_audit_path": str(update_audit_path),
        "sampling_box": (
            "no-flex AC power-flow reference +/- exact aggregate device "
            f"coordinate flexibility radii times {SAMPLING_EXPANSION_FACTOR:g}; "
            "no explicit loss margin"),
        "sampling_expansion_factor": SAMPLING_EXPANSION_FACTOR,
        "sampling_class_scarcity_threshold": (
            SAMPLING_CLASS_SCARCITY_THRESHOLD),
        "sampling_factor_changed_automatically": False,
        "sampling_box_used_by_update_optimization": False,
        "radial_metrics_used_during_training": False,
        "offline_timing_definition": (
            "sum over all independently trained conditions; includes initial "
            "reference-point calculation, physical label generation, "
            "polytope initialization, all learning "
            "fits, KKT/true-domain update-point generation, and update setup; "
            "excludes common test-truth preparation and final evaluation"),
        "offline_total_includes_learning": True,
        "offline_timing_checkpoint_safe": True,
        "legacy_initial_label_cache_reused": REUSE_LEGACY_INITIAL_DATA,
        "all_raw_evaluation_arrays_saved": True,
        "machine_signature": json.loads(shared.MACHINE_SIGNATURE),
    })
    print("\nExperiment completed")
    print(f"polygon library: {bank_path}")
    print(f"Complete direction-by-direction evaluation: {EVAL_DIR}")
    print(f"Summary: {summary_path}")


def retime_updated_model(stage="formal"):
    configure_stage(stage)
    bank_path = MODEL_DIR / "independent_polygon_bank.npz"
    if not bank_path.exists():
        raise FileNotFoundError(bank_path)
    with np.load(bank_path) as data:
        conditions = np.asarray(data["dtheta"], dtype=np.float32)
        A = np.asarray(data["A_updated"], dtype=np.float32)
        b = np.asarray(data["b_updated"], dtype=np.float32)
    model = ExactScenarioPolytopeBank(conditions, A, b).to(DEVICE)
    elapsed = measure_exact_lookup_once(model, conditions[0])
    summary_path = OUT / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Can’t find the official summary and can’t write back the online time: {summary_path}")
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    matched = False
    for row in rows:
        if row.get("method") == "lin_paper_updated":
            row["online_inference_seconds"] = repr(float(elapsed))
            matched = True
            break
    if not matched:
        raise ValueError(f"{summary_path}not found inlin_paper_updated.")
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    timing_path = OUT / "online_timing_original_protocol_lin_paper_updated.npz"
    np.savez(
        timing_path,
        method=np.asarray("lin_paper_updated"),
        online_inference_seconds=np.asarray(elapsed),
        timing_scope=np.asarray(
            "single_numpy_to_tensor_device_exact_scenario_lookup_A_b_to_cpu_numpy"),
        model_loading_included=np.asarray(False),
        warmup_included=np.asarray(False),
        repeats=np.asarray(1),
        timer=np.asarray("time.time"),
        machine_signature=np.asarray(shared.MACHINE_SIGNATURE),
    )
    print(f"[retime] paper Lin exact lookup: {1000.0*elapsed:.4f} ms")
    print(f"[retime] Written back: {summary_path}")
    print(f"[retime] Timing audit: {timing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=tuple(STAGE_CONFIGS), default="diagnostic",
        help="running phase; PyCharmWhen running directly without parameters, the default isdiagnostic")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retime-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="training sharded0base index, e.g.0to9")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of training shards")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true",
                      help="Just prepare50Public true feasible region of operating conditions, read-only for parallel sharding")
    mode.add_argument("--train-only", action="store_true",
                      help="Only train specified shards and do not write shared summary files")
    mode.add_argument("--finalize-only", action="store_true",
                      help="No training; read all completed results and evaluate them uniformly")
    args = parser.parse_args()
    if args.retime_only:
        retime_updated_model(args.stage)
    else:
        main(args.stage, args.overwrite,
             shard_index=args.shard_index, num_shards=args.num_shards,
             train_only=args.train_only, finalize_only=args.finalize_only,
             prepare_only=args.prepare_only)
