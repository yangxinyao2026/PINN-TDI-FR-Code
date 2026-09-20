# -*- coding: utf-8 -*-
"""Shared reference-point calculation for distribution-system cases."""

import copy
import io
import sys

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


REFERENCE_MEMBERSHIP_TOL = 1e-8


def reference_point_membership(A, b, x_ref, tol=REFERENCE_MEMBERSHIP_TOL):
    """Check whether ``x_ref`` belongs to the polytope ``A x <= b``.

    Returns ``(is_member, max_violation)``.  A non-finite input is treated as
    invalid instead of being allowed to propagate into a radial length.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    if (A.ndim != 2 or b.size == 0 or A.shape[0] != b.size or A.shape[1] != x_ref.size
            or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b))
            or not np.all(np.isfinite(x_ref))):
        return False, np.inf
    max_violation = float(np.max(A @ x_ref - b))
    return max_violation <= tol, max_violation


class ReferencePointCalculator:
    """Calculate the no-flexibility TDI reference point for one case.

    Single-phase cases use PYPOWER AC power flow.  Three-phase cases use a
    cached clone of the same Pyomo operating-region model, with every
    non-slack injection fixed to its measured load.
    """

    def __init__(self, ppc, init_params_dict, original_model=None, solver="ipopt"):
        self.ppc = ppc
        self.init_params_dict = init_params_dict
        self.pd_init = np.asarray(
            init_params_dict["Pd_meta"]["initial_value"], dtype=float
        )
        self.qd_init = np.asarray(
            init_params_dict["Qd_meta"]["initial_value"], dtype=float
        )
        self.is_3phase = self.pd_init.ndim == 2

        if self.pd_init.shape != self.qd_init.shape:
            raise ValueError("Pd_meta and Qd_meta must have the same shape")

        self._noflex_model = None
        self._noflex_solver = None
        if self.is_3phase:
            if original_model is None:
                raise ValueError("Three-phase reference points require original_model")
            self._build_3phase_noflex_model(original_model, solver)

    def compute(self, dtheta):
        """Return ``[P_TDI, Q_TDI]`` in per unit, or ``None`` on failure."""
        pd_meta, qd_meta = self._split_dtheta(dtheta)
        if self.is_3phase:
            return self._compute_3phase_noflex(pd_meta, qd_meta)
        return self._compute_single_phase_pypower(pd_meta, qd_meta)

    def _split_dtheta(self, dtheta):
        dtheta = np.asarray(dtheta, dtype=float).reshape(-1)
        expected = self.pd_init.size + self.qd_init.size
        if dtheta.size != expected:
            raise ValueError(
                f"dtheta has length {dtheta.size}, but {expected} values are required"
            )
        pd_meta = self.pd_init + dtheta[: self.pd_init.size].reshape(
            self.pd_init.shape
        )
        qd_meta = self.qd_init + dtheta[self.pd_init.size :].reshape(
            self.qd_init.shape
        )
        return pd_meta, qd_meta

    def _compute_single_phase_pypower(self, pd_meta, qd_meta):
        from pypower.api import ppoption, runpf

        base_mva = self.ppc["baseMVA"]
        ppc_copy = copy.deepcopy(self.ppc)
        ppc_copy["gen"] = ppc_copy["gen"].astype(float)

        pd_nominal_pu = self.ppc["bus"][:, 2] / base_mva
        qd_nominal_pu = self.ppc["bus"][:, 3] / base_mva
        new_pd_pu = (0.8 + 0.4 * pd_meta) * pd_nominal_pu
        new_qd_pu = (0.8 + 0.4 * qd_meta) * qd_nominal_pu
        ppc_copy["bus"][:, 2] = new_pd_pu * base_mva
        ppc_copy["bus"][:, 3] = new_qd_pu * base_mva

        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            results, success = runpf(ppc_copy, ppopt)
        except Exception:
            results, success = None, False
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        if not success:
            return None
        return np.array(
            [results["gen"][0, 1] / base_mva, results["gen"][0, 2] / base_mva],
            dtype=float,
        )

    def _build_3phase_noflex_model(self, original_model, solver):
        model = original_model.clone()

        for objective in model.component_data_objects(pyo.Objective, active=True):
            objective.deactivate()

        model.noflex_constraints = pyo.ConstraintList()
        bus_ids = list(model.BUS)
        slack_bus = 1 if 1 in bus_ids else bus_ids[0]
        node_flex_dict = self.ppc.get("node_flex_dict", {})
        for bus in bus_ids:
            if bus == slack_bus:
                continue
            flex_info = node_flex_dict.get(bus, {"type": 0})
            if not flex_info or not flex_info.get("type", 0):
                # The original model already fixes type-0 injections exactly.
                continue
            for phase in model.PH:
                model.noflex_constraints.add(model.Pn[bus, phase] == -model.Pd[bus, phase])
                model.noflex_constraints.add(model.Qn[bus, phase] == -model.Qd[bus, phase])

        # The no-flexibility power-flow equations determine var_proj.  A small
        # loss objective selects the normal high-voltage solution if the
        # nonlinear equations admit multiple mathematical roots.
        model.reference_objective = pyo.Objective(
            expr=sum(model.I2[index] for index in model.I2), sense=pyo.minimize
        )
        self._noflex_model = model
        self._noflex_solver = SolverFactory(solver, tee=False)

    def _compute_3phase_noflex(self, pd_meta, qd_meta):
        model = self._noflex_model
        bus_ids = list(model.BUS)
        phases = list(model.PH)
        if pd_meta.shape != (len(bus_ids), len(phases)):
            raise ValueError(
                "Three-phase Pd_meta/Qd_meta shape does not match model.BUS x model.PH"
            )

        for i, bus in enumerate(bus_ids):
            for j, phase in enumerate(phases):
                model.Pd_meta[bus, phase] = float(pd_meta[i, j])
                model.Qd_meta[bus, phase] = float(qd_meta[i, j])

        try:
            result = self._noflex_solver.solve(model)
        except Exception:
            return None
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            return None
        return np.array([pyo.value(model.var_proj[j]) for j in range(2)], dtype=float)
