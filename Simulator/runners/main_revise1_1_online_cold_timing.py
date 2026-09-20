# -*- coding: utf-8 -*-
"""Measure first-call online inference times for the R1.1, R1.6, and PINN methods."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

from Simulator.Approximator import FullNet, PreTrainNet


CASE_NAME = "case33bw_ds"
LABEL_SIZES = (50, 100, 200, 500, 1000, 2000, 5000)
PROTOCOL_VERSION = "r11_r16_manuscript_single_region_protocol_v4"


def publication_safe_path(path: Path) -> str:
    """Return a repository-relative path without exposing the local machine."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name

SUPERVISED_ROOT = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "supervised" / CASE_NAME
)
SUPPORT_ROOT = SUPERVISED_ROOT / "support_points"
LABEL_PATH = SUPPORT_ROOT / "labels" / "train_labels.npz"
PINN_WEIGHTS = (
    SUPERVISED_ROOT / "models" / "pinn_retrained"
    / "fullnet_weights_feasible.pth"
)
PINN_PRETRAIN_WEIGHTS = (
    SUPERVISED_ROOT / "models" / "pinn_retrained"
    / "pretrainnet_weights.pth"
)
SUPERVISED_WEIGHTS = {
    f"supervised_n{n}": (
        SUPPORT_ROOT / "models" / f"n{n}" / "supervised_fullnet_weights.pth"
    )
    for n in LABEL_SIZES
}

LEARNING_ROOT = (
    PROJECT_ROOT / "results" / "ds_proj_revise_V1" / "comparison"
    / "learning_methods" / CASE_NAME
)
R11_BANKS = {
    "r11_ac_validated_updated": (
        LEARNING_ROOT / "lin_paper_2d_formal_validated_timed"
        / "models" / "independent_polygon_bank.npz"
    ),
}
METHODS = (
    "pinn",
    *(f"supervised_n{n}" for n in LABEL_SIZES),
    *R11_BANKS,
)
OUTPUT_DIR = LEARNING_ROOT / "online_manuscript_protocol_current_hardware"


class BoundaryPointNet(nn.Module):
    """with currentR1.6same66->128->72network of support points."""

    def __init__(self, dim_theta: int, points_init: np.ndarray,
                 device: torch.device):
        super().__init__()
        points_init = np.asarray(points_init, dtype=np.float32)
        if points_init.shape != (36, 2):
            raise ValueError(f"The support point initialization should be(36,2), Actually{points_init.shape}")
        self.net = nn.Sequential(
            nn.Linear(dim_theta, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 72, device=device),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.as_tensor(
                points_init.reshape(-1), dtype=torch.float32, device=device))

    def forward(self, delta_theta: torch.Tensor) -> torch.Tensor:
        return self.net(delta_theta).reshape(-1, 36, 2)


class ExactScenarioPolytopeBank(nn.Module):
    """R1.1Accurate trained operating condition lookup table, nearest neighbor extrapolation is not allowed."""

    def __init__(self, conditions: np.ndarray, A: np.ndarray, b: np.ndarray,
                 device: torch.device, match_tol: float = 1e-6):
        super().__init__()
        self.match_tol = float(match_tol)
        self.register_buffer("conditions", torch.as_tensor(
            conditions, dtype=torch.float32, device=device))
        self.register_buffer("A_all", torch.as_tensor(
            A, dtype=torch.float32, device=device))
        self.register_buffer("b_all", torch.as_tensor(
            b, dtype=torch.float32, device=device))

    def forward(self, theta: torch.Tensor):
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        error = torch.amax(torch.abs(
            theta[:, None, :] - self.conditions[None, :, :]), dim=2)
        value, index = torch.min(error, dim=1)
        if bool(torch.any(value > self.match_tol)):
            raise ValueError("R1.1The polygon library receives new cases that were not independently trained.")
        return self.A_all[index], self.b_all[index]


def convex_hull_polygon(points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """withR1.6same caliber, will36prediction points are constructed into a counterclockwise convex hull."""
    points = np.asarray(points, dtype=float)
    if points.shape != (36, 2) or not np.all(np.isfinite(points)):
        raise ValueError(f"Invalid supervision network output: {points.shape}")
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    unique = []
    for point in ordered:
        if not unique or np.linalg.norm(point - unique[-1]) > tol:
            unique.append(point.copy())
    if len(unique) < 3:
        raise RuntimeError("Predicted points are not sufficient to form a 2D convex hull.")

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

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
        raise RuntimeError("Predicting point convex hull degradation.")
    return hull


def required_files() -> dict[str, Path]:
    files = {
        "supporttraining label": LABEL_PATH,
        "PINNPre-trained weights": PINN_PRETRAIN_WEIGHTS,
        "PINNweight": PINN_WEIGHTS,
    }
    files.update({f"{method}weight": path
                  for method, path in SUPERVISED_WEIGHTS.items()})
    files.update({f"{method}polygon library": path
                  for method, path in R11_BANKS.items()})
    return files


def preflight(require_cuda: bool = True) -> None:
    missing = [f"{name}: {path}" for name, path in required_files().items()
               if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files required for unified online timing: \n" + "\n".join(missing))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "Not detected by the current interpreterCUDA; Official timing must be used in trainingGPUServer running.")


def load_label_basis() -> tuple[int, np.ndarray]:
    with np.load(LABEL_PATH, allow_pickle=False) as data:
        theta = np.asarray(data["dtheta"], dtype=np.float32)
        points_init = np.asarray(data["support_points"][0], dtype=np.float32)
    return int(theta.shape[1]), points_init


def manuscript_cuda_initialization(device: torch.device):
    """reproduce original ``load_fullnet`` untimed PreTrainNet forward."""
    pretrain = PreTrainNet(
        np.zeros((36, 2), dtype=np.float32),
        np.zeros(36, dtype=np.float32),
        is_epigraph=False,
        device=device,
    )
    pretrain.load_state_dict(torch.load(
        PINN_PRETRAIN_WEIGHTS, map_location=device, weights_only=True))
    pretrain = pretrain.to(device)
    pretrain.eval()
    with torch.no_grad():
        A_pre, b_pre = pretrain()
    # The original manuscript is then converted intoCPU numpy; This step will waitCUDAComplete.
    A_pre = A_pre[0].detach().cpu().numpy()
    b_pre = b_pre[0].detach().cpu().numpy()
    return A_pre, b_pre


def load_pinn(device: torch.device, pretrained_basis=None):
    dim_theta, _ = load_label_basis()
    if pretrained_basis is None:
        pretrained_basis = manuscript_cuda_initialization(device)
    A_init, b_init = pretrained_basis
    model = FullNet(
        dim_theta=dim_theta, A_init=A_init, b_init=b_init,
        hidden_sizes=[128], activation="relu", device=device,
    ).to(device)
    model.load_state_dict(torch.load(
        PINN_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    return model, np.zeros(dim_theta), PINN_WEIGHTS, "A_b"


def load_supervised(method: str, device: torch.device):
    dim_theta, points_init = load_label_basis()
    model = BoundaryPointNet(dim_theta, points_init, device).to(device)
    weights = SUPERVISED_WEIGHTS[method]
    model.load_state_dict(torch.load(
        weights, map_location=device, weights_only=True))
    model.eval()
    return (model, np.zeros(dim_theta), weights,
            "support_points_convex_hull")


def load_r11(method: str, device: torch.device):
    bank_path = R11_BANKS[method]
    with np.load(bank_path, allow_pickle=False) as data:
        conditions = np.asarray(data["dtheta"], dtype=np.float32)
        A = np.asarray(data["A_updated"], dtype=np.float32)
        b = np.asarray(data["b_updated"], dtype=np.float32)
    model = ExactScenarioPolytopeBank(conditions, A, b, device)
    model.eval()
    return model, conditions[0].astype(np.float64), bank_path, "exact_A_b_lookup"


def load_method(method: str, device: torch.device, pretrained_basis=None):
    if method == "pinn":
        return load_pinn(device, pretrained_basis=pretrained_basis)
    if method.startswith("supervised_n"):
        return load_supervised(method, device)
    if method in R11_BANKS:
        return load_r11(method, device)
    raise ValueError(f"unknown method: {method}")


def online_region_call(model, dtheta: np.ndarray, output_kind: str,
                       device: torch.device) -> dict[str, list[int]]:
    """Perform a full zone generation consistent with official online use."""
    tensor = torch.as_tensor(dtheta, dtype=torch.float32, device=device)
    output = model(tensor)
    if output_kind == "support_points_convex_hull":
        points = output.detach().cpu().numpy()[0]
        region = convex_hull_polygon(points)
        return {
            "predicted_support_points": list(points.shape),
            "convex_hull_vertices": list(region.shape),
        }
    A, b = output
    A_cpu = A.detach().cpu().numpy()
    b_cpu = b.detach().cpu().numpy()
    if not (np.all(np.isfinite(A_cpu)) and np.all(np.isfinite(b_cpu))):
        raise RuntimeError("Online area output contains non-finite values.")
    return {"A": list(A_cpu.shape), "b": list(b_cpu.shape)}


def measure_once(method: str, output_path: Path) -> None:
    """Complete online area output after one model load, timed in original order."""
    preflight(require_cuda=True)
    device = torch.device("cuda")
    # Run the untimed PreTrainNet forward pass before constructing FullNet for every method.
    # CUDAInitialization to avoid independent child processesCUDAContext startup time is only counted for a certain method.
    pretrained_basis = manuscript_cuda_initialization(device)
    model, dtheta, source, output_kind = load_method(
        method, device, pretrained_basis=pretrained_basis)

    with torch.inference_mode():
        # The original manuscript is not correctFullNetPerform the same call to warm up and usetime.time()Timing.
        # output toCPU numpyWill wait for this timeCUDAforward truly complete.
        started = time.time()
        output_shapes = online_region_call(
            model, dtheta, output_kind, device)
        elapsed = time.time() - started

    family = ("PINN" if method == "pinn" else
              "supervised_boundary_point_network"
              if method.startswith("supervised_n") else
              "R1.1_exact_trained_scenario_polygon_lookup")
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "method_family": family,
        "online_seconds": float(elapsed),
        "online_milliseconds": float(1000.0 * elapsed),
        "output_kind": output_kind,
        "output_shapes": output_shapes,
        "dim_theta": int(len(dtheta)),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": "anonymized-host",
        "source": publication_safe_path(source),
        "model_loading_included": False,
        "warmup_forward_calls": 0,
        "untimed_pretrain_initialization_calls": 1,
        "timed_online_calls": 1,
        "timer": "time.time",
        "timing_scope": (
            "manuscript order: untimed PreTrainNet CUDA initialization, then "
            "first timed numpy-to-CUDA target-model forward and CPU numpy output; supervised "
            "methods additionally include CPU convex-hull construction"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{method}] Original process current hardware retest: {1000.0 * elapsed:.6f} ms")


def run_child(method: str, output_path: Path) -> None:
    command = [
        sys.executable, "-m",
        "Simulator.runners.main_revise1_1_online_cold_timing",
        "--child", method, "--child-output", str(output_path),
    ]
    print("\n" + "=" * 78)
    print("Independently timed subprocesses: " + method)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def save_summary(records: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "online_manuscript_protocol_times.csv"
    fields = [
        "method", "method_family", "online_seconds", "online_milliseconds",
        "output_kind", "device", "hostname", "torch_version",
        "warmup_forward_calls", "untimed_pretrain_initialization_calls",
        "timed_online_calls",
        "model_loading_included", "timing_scope", "source",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)

    metadata_path = OUTPUT_DIR / "experiment_metadata.json"
    metadata_path.write_text(json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "case": CASE_NAME,
        "measurement": (
            "one fresh independent process per method; one untimed manuscript "
            "PreTrainNet initialization followed by the first timed target-model call"),
        "warmup_forward_calls_per_method": 0,
        "untimed_pretrain_initialization_calls_per_method": 1,
        "timed_online_calls_per_method": 1,
        "model_loading_included": False,
        "r16_summary_and_raw_timing_synchronized": True,
        "methods": list(METHODS),
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Unified timingCSVIt is the only data source; it is synchronized after each retest.R1.6There are summaries andrawFile,
    # Avoid the recurrence of problems arising from the evaluation process1 mstime after preheating.
    from Simulator.runners import main_revise1_6_supervised as r16
    r16.synchronize_existing_online_times(required=True)

    print("\n" + "=" * 78)
    print("R1.1/R1.6The current hardware retest of the original manuscript process is completed")
    print("-" * 78)
    for row in records:
        print(f"{row['method']:<36s}{row['online_milliseconds']:>14.6f} ms")
    print("-" * 78)
    print("Model loading andPreTrainNetThe initialization is not timed; the target network is not warmed up; the first complete output is timed.1times.")
    print(f"result: {csv_path}")
    print(f"metadata: {metadata_path}")


def main(check_only: bool = False) -> None:
    preflight(require_cuda=True)
    print(f"Python: {sys.executable}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print("Caliber: independent subprocess; originalPreTrainNetThe initialization is not timed; the target network is not warmed up; the first complete area output is timed..")
    if check_only:
        print(f"Check passed: {len(required_files())}all required files are present.")
        return

    run_dir = OUTPUT_DIR / "individual_runs"
    records = []
    for method in METHODS:
        output_path = run_dir / f"{method}.json"
        run_child(method, output_path)
        records.append(json.loads(output_path.read_text(encoding="utf-8")))
    save_summary(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Only check all weights, polygon library andCUDA.")
    parser.add_argument("--child", choices=METHODS, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is not None:
        if args.child_output is None:
            parser.error("--childNeed to be provided at the same time--child-output")
        measure_once(args.child, args.child_output)
    else:
        main(check_only=args.check_only)
