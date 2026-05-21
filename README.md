# PINN-TDI-Code

Code and data for **Physics-Informed Neural Network-based Two-stage Decomposition Iteration (PINN-TDI)** — a method for approximating parametric feasible regions of power system optimization problems using neural networks.

## Overview

This project implements a two-stage neural network training framework that learns polyhedral approximations of feasible regions from complex power system optimization models. The key idea is:

1. **PreTrainNet** — learns a baseline polyhedral approximation `{x : Ax ≤ b}` for the feasible region at nominal parameter values.
2. **FullNet / BiasNet** — takes parameter perturbations `Δθ` as input and outputs adjusted polyhedron matrices `(A(θ), b(θ))`, enabling the approximation to adapt to varying operating conditions.

The approach supports both convex (polygon, ellipse) and non-convex feasible regions, and scales to realistic distribution system optimal power flow (OPF) models.

## Project Structure

```
share/
├── Simulator/
│   ├── Approximator.py          # Core: neural networks, error calculator, trainer
│   ├── Counter.py               # Hausdorff distance calculator (polyhedron vs. ball)
│   ├── Plotter.py               # 2D/3D visualization and error analysis tools
│   ├── __init__.py              # Project root path
│   ├── cases/
│   │   ├── basic_cases.py       # Synthetic cases: polygon, ellipse, epigraph, nonconvex
│   │   ├── TD_case.py           # Transmission + distribution system OPF models
│   │   └── DS_case_3phase.py    # Three-phase unbalanced distribution system model
│   ├── runners/
│   │   ├── main_polygon.py      # Run polygon case training
│   │   ├── main_ellipse.py      # Run ellipse case training
│   │   └── main_ds.py           # Run distribution system case training
│   ├── draw_pictures/
│   │   └── main_ds_plot.py      # Plot training error from saved data
│   ├── testers/
│   │   └── tst_TD.py            # T&D integrated case tests
│   └── data/
│       ├── TD_OPF/              # IEEE and custom test case data (.mat, .m)
│       ├── profiles_data/       # Load/PV/Baseline profile data
│       └── real_dis_data/       # Real distribution system measurement data
└── results/                     # Trained model weights and figures
```

## Key Components

### Approximator.py

- **`ErrorCalculator`** — Evaluates feasibility and optimality errors between the original feasible region and its polyhedral approximation via optimization-based projection.
- **`PreTrainNet`** — Neural network that outputs `(A, b)` with no parameter input (learns a fixed approximation).
- **`BiasNet`** — Neural network that outputs only `b(θ)` while keeping `A` fixed from pretraining.
- **`FullNet`** — Neural network that outputs both `A(θ)` and `b(θ)` conditioned on parameter perturbations.
- **`Trainer`** — Training loop supporting SGD/Adam, learning rate scheduling, batch training, and optional parallel computation.
- **`compute_loss`** — Vectorized loss function based on active constraint violations.

### Counter.py

- **`PolyBallHausdorffCalculator`** — Computes the Hausdorff distance between a polyhedron `{x : Ax ≤ b}` and a unit ball, with optional sensitivity analysis via perturbation.

### Plotter.py

- **`ShapeDrawer_2D`** — Draws polygons, ellipses, epigraphs, and non-convex regions in 2D.
- **`ShapeDrawer_3D`** — Visualizes the evolution of polyhedral approximations across training iterations as 3D prisms.
- **`ErrorVisualizer`** — Box plots, violin plots, and KDE plots for error distribution analysis.

### Cases

| Case | Description | Solver |
|------|-------------|--------|
| `polygon` | 2D octagon with adjustable constraints | Gurobi |
| `ellipse` | 2D ellipse defined by quadratic form | Gurobi |
| `epigraph` | Epigraph-form problem with upper bound | Ipopt |
| `nonconvex` | Unit circle minus inner circle (non-convex) | Ipopt |
| `ball` / `cube` | High-dimensional ball/cube benchmarks | Gurobi |
| `case33bw_ds` | IEEE 33-bus radial distribution system | Ipopt |
| `case118zh_ds` | 118-bus distribution system | Ipopt |
| `case533mt_hi_ds` | 533-bus distribution system | Ipopt |
| `case36real_3phase_ds` | Real 36-node three-phase unbalanced system | Ipopt |
| T&D integrated | Combined transmission-distribution systems | Ipopt |

## Requirements

- Python 3.12+
- [Pyomo](https://pyomo.readthedocs.io/) — optimization modeling
- [PyTorch](https://pytorch.org/) — neural network training
- [Gurobi](https://www.gurobi.com/) — LP/QP solver (for basic cases)
- [Ipopt](https://github.com/coin-or/Ipopt) — NLP solver (for distribution system cases)
- NumPy, SciPy, Matplotlib, Pandas, Pillow

## Usage

### 1. Basic Cases (Polygon / Ellipse)

```bash
# PreTrainNet stage
python Simulator/runners/main_polygon.py   # set model_type = 'pretrainnet'

# FullNet stage (loads pretrained weights)
python Simulator/runners/main_polygon.py   # set model_type = 'fullnet'
```

Edit `model_type` and `parallel` at the top of each runner script to switch between training modes.

### 2. Distribution System Cases

```bash
python Simulator/runners/main_ds.py
```

Configure the case in the `dscases` dictionary inside `main_ds.py`. Available cases include `case33bw_ds`, `case118zh_ds`, `case533mt_hi_ds`, and `case36real_3phase_ds`.

### 3. Plot Training Errors

After training with `record_errors = True`:

```bash
python Simulator/draw_pictures/main_ds_plot.py
```

### 4. Two-Stage Training Pipeline

The distribution system training follows a two-stage protocol:

1. **Stage 1 (PreTrainNet):** Trains for ~2000 iterations to learn a baseline polyhedral approximation.
2. **Stage 2 (FullNet):** Fine-tunes both `A` and `b` conditioned on load parameter perturbations, then optionally minimizes feasibility error with a smaller learning rate.

## Method

Given an original feasible region `Ω(θ) = {x : g(x, θ) ≤ 0}` parameterized by `θ`, the method learns a polyhedral approximation `P(θ) = {x : A(θ)x ≤ b(θ)}` that minimizes:

- **Feasibility error:** Distance from points in `P(θ)` to `Ω(θ)`
- **Optimality error:** Distance from points in `Ω(θ)` to `P(θ)`

The neural network is trained using gradient-based optimization with a custom loss that penalizes active constraint violations, evaluated through repeated calls to mathematical programming solvers.

## Data

- **`Simulator/data/TD_OPF/`** — Standard IEEE test case data in MATLAB format
- **`Simulator/data/profiles_data/`** — Load and PV generation profiles
- **`Simulator/data/real_dis_data/`** — Real measurement data from a 36-node three-phase distribution network

## Results

Trained model weights (`.pth`), error data (`.npz`), and visualization figures are saved under `results/ds_proj_paper/<case_name>/`.
