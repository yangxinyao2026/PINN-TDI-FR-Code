# -*- coding: utf-8 -*-
"""Create and validate the shared case33 test conditions used by R1.1 and R1.6."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from Simulator import PROJECT_ROOT


CASENAME = "case33bw_ds"
CONDITION_COUNT = 50
DIM_THETA = 66                  # case33of33aPd_metaand33aQd_meta.
DTHETA_RANGE = (-0.5, 0.5)     # withPINNThe sampling range of operating conditions is consistent.
CONDITION_SEED = 44             # Use the originalR1.6test dividedSEED+2.
CONDITION_PATH = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "common_conditions" / CASENAME / "test_conditions.npz"
)
METHOD_VERSION = "case33_common_test_conditions_v1"


def ensure_common_test_conditions(dim_theta=DIM_THETA, overwrite=False):
    """Generate or read fixed conditions; boundary labels and true domains are not calculated here."""
    dim_theta = int(dim_theta)
    if CONDITION_PATH.exists() and not overwrite:
        with np.load(CONDITION_PATH) as data:
            conditions = np.asarray(data["dtheta"], dtype=np.float64)
            valid = (
                conditions.shape == (CONDITION_COUNT, dim_theta)
                and str(data.get("method_version", "")) == METHOD_VERSION
                and int(data.get("seed", -1)) == CONDITION_SEED
                and np.allclose(
                    np.asarray(data.get("dtheta_range", []), dtype=float),
                    DTHETA_RANGE, rtol=0.0, atol=0.0)
            )
        if valid:
            return CONDITION_PATH
        raise ValueError(
            f"The public operating condition file is inconsistent with the current agreement: {CONDITION_PATH}; "
            "Please confirm before use--overwriteRegenerate.")

    rng = np.random.RandomState(CONDITION_SEED)
    conditions = rng.uniform(
        DTHETA_RANGE[0], DTHETA_RANGE[1],
        size=(CONDITION_COUNT, dim_theta))
    CONDITION_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CONDITION_PATH,
        dtheta=conditions,
        count=np.asarray(CONDITION_COUNT),
        dim_theta=np.asarray(dim_theta),
        dtheta_range=np.asarray(DTHETA_RANGE, dtype=float),
        seed=np.asarray(CONDITION_SEED),
        casename=np.asarray(CASENAME),
        purpose=np.asarray("shared formal evaluation conditions for R1.1 and R1.6"),
        method_version=np.asarray(METHOD_VERSION),
    )
    print(f"Generated50public operating conditions: {CONDITION_PATH}")
    return CONDITION_PATH


def load_common_test_conditions(dim_theta=DIM_THETA, overwrite=False):
    path = ensure_common_test_conditions(dim_theta, overwrite)
    with np.load(path) as data:
        return np.asarray(data["dtheta"], dtype=np.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_common_test_conditions(overwrite=args.overwrite)
