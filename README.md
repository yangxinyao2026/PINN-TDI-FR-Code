# Distribution-Network Feasible-Region Learning: Revision Experiments

This repository contains the code from the original manuscript, additional experiments conducted during peer review, and the corresponding final results. `results/ds_proj_revise_V1/` is the root directory for the revision results, while `results/ds_proj_paper/` preserves the original-manuscript results.

## Repository structure

```text
share/
├── Simulator/
│   ├── Approximator.py               # Core PINN, supervised-network, and error calculations
│   ├── solver_environment.py         # Solver environment configuration
│   ├── reference_point.py            # Reference operating points and membership checks
│   ├── cases/                        # Distribution, transmission, and analytical test cases
│   ├── data/                         # Test-system and time-series data
│   ├── runners/                      # Main experiment entry points
│   ├── draw_pictures/                # Manuscript and diagnostic plotting scripts
│   ├── draw_compact/                 # Standalone compact publication-figure scripts
│   └── testers/                      # Small-scale validation scripts
├── results/
│   ├── ds_proj_paper/                # Original-manuscript results
│   ├── ds_proj_revise_V1/
│       ├── ablation/                 # Ablation experiments
│       ├── comparison/               # Peer-review comparison experiments
│       ├── synthetic_star/           # Analytical star-shaped nonconvex-domain experiment
│       └── synthetic_two_disks/      # Two-disk-union nonconvex-domain experiment
│   └── pictures_compact/             # Compact PDF panels generated from saved results
└── README.md
```

## Data-release statement

The 36-bus three-phase test case is provided in `Simulator/data/TD_OPF/ds_data/case36real_3phase_ds.mat`. The file contains only the final bus, branch, three-phase load, and three-phase impedance values directly consumed by the model. Raw collection tables, geographic identifiers, customer identifiers, conductor types, line lengths, sampling timestamps, and source mappings are not part of the public release.

## Main experiment entry points

Run the following commands from the project root (`share`). The corresponding scripts can also be run directly from an IDE.

### R1.1: two-module learning method used in the manuscript

The retained workflow uses data updates independently verified against the AC power-flow physics. The 50 formal operating conditions are split into 10 shards, with five conditions per shard:

```powershell
python -m Simulator.runners.main_revise1_1_learning_comparison_prepare
python -m Simulator.runners.main_revise1_1_learning_comparison_shard01
# Run shard02 through shard10 in parallel.
python -m Simulator.runners.main_revise1_1_learning_comparison_finalize
```

### R1.6: PINN versus support-point supervised learning

The supervised labels are ordered boundary points obtained by solving AC feasible-region support optimizations in 36 fixed unit directions. Training-set sizes are `50/100/200/500/1000/2000/5000`; validation and testing each use 50 independent operating conditions.

```powershell
python -m Simulator.runners.main_revise1_6_supervised --stage labels
python -m Simulator.runners.main_revise1_6_supervised --stage pinn
python -m Simulator.runners.main_revise1_6_supervised --stage train
python -m Simulator.runners.main_revise1_6_supervised --stage evaluate
```

Running the script without arguments executes the complete formal workflow. Existing labels and models are reused when their protocol metadata match; add `--overwrite` explicitly to recompute them. The final R1.1, R1.6, and PINN evaluations share the 50 test conditions in `results/ds_proj_revise_V1/comparison/common_conditions/`.

### Unified online timing

```powershell
python -m Simulator.runners.main_revise1_1_online_cold_timing
```

Each method is timed in an independent subprocess. Model loading and `PreTrainNet` initialization are excluded, the target network is not warmed up, and only the first complete online region output is timed. The sole formal timing file is:

```text
results/ds_proj_revise_V1/comparison/learning_methods/case33bw_ds/
online_manuscript_protocol_current_hardware/online_manuscript_protocol_times.csv
```

After timing, the program automatically synchronizes the R1.6 `summary.csv` and raw evaluation file.

### R2.5: near-convexity and analytical nonconvex cases

```powershell
# Support-point/chord near-convexity tests for case33, case36, case118, and case533
python -m Simulator.runners.main_revise2_5_support_chord_convexity

# Analytical nonconvex-domain comparisons; existing results are protected by default
python -m Simulator.runners.main_revise2_5_star --overwrite --reuse-pretrain
python -m Simulator.runners.main_revise2_5_two_disks --overwrite --reuse-pretrain
```

The `star` and `two_disks` runners support `--help`, `--overwrite`, `--reuse-pretrain`, and an optional `--seed`. Coverage continues to use the maximum radial length, consistently with `Simulator/draw_pictures/approximate_polygon_coverage.py`.

### Other revision experiments

The principal entry points are:

- `main_revise1_2_amortization.py`: offline-cost amortization;
- `main_revise1_3_thermal.py`: thermal constraints;
- `main_revise1_5and3_2downstream.py` and `main_revise1_5and3_2downstream_large.py`: downstream-system experiments;
- `main_revise1_7_topology.py`: topology changes;
- `main_revise2_3_network.py`: neural-network architecture comparison;
- `main_revise2_4_uncertainty.py`: measurement uncertainty.

The large downstream runner contains four fixed 27-DSO configurations:

```powershell
# Solver-time protocol for all four configurations (disaggregation excluded)
python -m Simulator.runners.main_revise1_5and3_2downstream_large --version all --mode timing

# Full validation, including disaggregation, for one selected configuration
python -m Simulator.runners.main_revise1_5and3_2downstream_large --version v1 --mode formal
```

The version mapping is `v1=case33`, `v2=case533`, `v3=case118` with the
recorded interface scaling, and `v4=the anonymized three-phase case`. Full
validation is selected one version at a time so that its outputs and metadata
cannot be mixed with another configuration.

### Compact publication figures

The scripts in `Simulator/draw_compact/` read the committed numerical results
and export standalone PDF panels to `results/pictures_compact/`. They do not
train or modify any model. Run them from the repository root as Python modules:

| Script | Output subdirectory |
|---|---|
| `learning_comparison_compact.py` | `learning_comparison/` |
| `plot_nn_architecture_distributions.py` | `nn_architecture_distributions/` |
| `nonconvex_region_comparison.py` | `nonconvex_comparison/` |
| `R2_5_support_chord_near_convexity.py` | `support_chord_near_convexity/` |
| `R1_3_thermal_plot.py` | `thermal_comparison/` |
| `R1_7_topology_regions_plot.py` | `topology/` |
| `uncertainty_distribution_plot.py` | `uncertainty/` |

For example:

```powershell
python -m Simulator.draw_compact.learning_comparison_compact
python -m Simulator.draw_compact.R2_5_support_chord_near_convexity
```

## Experiment-to-result mapping

### Original-manuscript results

| Result directory | Associated code | Contents |
|---|---|---|
| `results/ds_proj_paper/` | `Simulator/runners/main_ds.py` and the original plotting scripts | Models, training records, errors, and figures for the distribution-network cases in the original manuscript. These are baseline inputs for the revision experiments and should not be overwritten. |

### Revision comparison experiments

All paths below are relative to `results/ds_proj_revise_V1/comparison/`.

| Result subdirectory | Main runner | Main plotting script | Principal outputs |
|---|---|---|---|
| `amortization/` | `main_revise1_2_amortization.py` | `R1_2_amortization_plot.py` | `summary.csv/.npz`, original online times, hardware metadata, and amortization curves. |
| `common_conditions/case33bw_ds/` | `common_case33_test_conditions.py` | — | `test_conditions.npz`, containing the 50 shared test conditions used for fair R1.1, R1.6, and PINN evaluation. |
| `downstream/case4gs_ts_case33bw_ds/` | `main_revise1_5and3_2downstream.py` | `R1_5and3_2_downstream_plot.py` | Operating-condition grid, per-condition solutions, and summaries for the small transmission-distribution coordination experiment. |
| `downstream/case118_ts_multi_case33bw_ds/` | `main_revise1_5and3_2downstream_large.py` | `R1_5and3_2_downstream_large_plot.py` | Earlier 10-DSO validation results for the 118-bus host system. |
| `downstream/case118_ts_equal_count_no_alpha/` | `main_revise1_5and3_2downstream_large.py` | `R1_5and3_2_downstream_large_plot.py` | Four equal-count experiments with 27 DSOs: case33, case533, case118 with interface scaling, and the anonymized three-phase case. The committed `solver_timing_27dso_5profiles_1load/` directories contain solver-timing runs; use `--mode formal` when disaggregation evidence is also required. |
| `learning_methods/case33bw_ds/lin_paper_2d_formal_validated_timed/` | R1.1 preparation, 10 shard runners, and finalization | `R1_1_learning_comparison_plot.py`, `R1_1_predicted_region_vs_sample_hull.py` | Per-condition polygon library, updated data, raw evaluation data, and `summary.csv` for the AC-verified R1.1 workflow. |
| `learning_methods/case33bw_ds/online_manuscript_protocol_current_hardware/` | `main_revise1_1_online_cold_timing.py` | Shared by the R1.1 and R1.6 plotting scripts | Unified manuscript-protocol online times for R1.1, the R1.6 supervised network, and the PINN. |
| `network/case33bw_ds/` | `main_revise2_3_network.py` | `R2_3_network_comparison_plot.py` | Weights, evaluation data, and `formal_summary.csv/.npz` for different hidden-layer and activation-function configurations. |
| `supervised/case33bw_ds/support_points/` | `main_revise1_6_supervised.py` | `R1_6_supervised_plot.py` | Labels, weights, directional evaluations, and `summary.csv` for the PINN and seven supervised-label set sizes. |
| `supervised/case36real_3phase_ds/radial_vs_support_boundary/` | `main_revise1_6_case36_region_comparison.py` | Generated directly by the runner | Comparison between the radial boundary and the 360-direction support-point boundary for the case36 three-phase system. |
| `supervised/radial_scan_step_validation/` | `validate_radial_scan_step.py` | — | Radial-scan step-size diagnostics for case33 and case36; these data are not formal R1.6 supervised labels. |
| `support_chord_convexity/` | `main_revise2_5_support_chord_convexity.py` | `R2_5_support_chord_near_convexity.py` | 360 support directions, chord-projection checks, pointwise data, summaries, and near-convexity figures for case33, case36, case118, and case533. |
| `thermal/case33bw_ds/` | `main_revise1_3_thermal.py` | `R1_3_thermal_plot.py`, `R1_3_thermal_region_preview_plot.py` | Baseline and 120% thermal-constraint experiments, calibration data, and revision evidence summaries. |
| `topology/case33bw_ds/` | `main_revise1_7_topology.py` | `R1_7_topology_plot.py` | Models, errors, coverage rates, summaries, and comparison figures for the Base and T1–T5 topologies. |
| `uncertainty/case33bw_ds/` | `main_revise2_4_uncertainty.py` | `R2_4_uncertainty_plot.py` | Raw squared errors under measurement noise, per-condition reference-point normalization, summaries, and figures. |
| `uncertainty/case10ba_ds/` | Same as above | Same as above | Measurement-uncertainty results for case10. |

### Analytical nonconvex-domain experiments

| Result directory | Associated code | Principal outputs |
|---|---|---|
| `results/ds_proj_revise_V1/synthetic_star/` | `main_revise2_5_star.py` | Projection/radial-method weights, training-evolution figures, `coverage_raw.npz`, and `comparison_metrics.csv` for the star-shaped nonconvex domain. |
| `results/ds_proj_revise_V1/synthetic_two_disks/` | `main_revise2_5_two_disks.py` | Projection/radial-method weights, training-evolution figures, `coverage_raw.npz`, and `comparison_metrics.csv` for the union of two disks. |

### Ablation directory

`results/ds_proj_revise_V1/ablation/` currently contains only `.gitkeep`; no formal ablation data have been saved yet.

### Result-file conventions

- `summary.csv`, `formal_summary.csv`, and `comparison_metrics.csv`: summary results used first for manuscript tables or aggregate comparisons;
- `*.npz`: complete numerical arrays for per-condition or per-direction errors, coverage rates, labels, and model evaluations;
- `models/` and `*.pth`: trained neural-network weights or per-condition polygon libraries;
- `experiment_config.json`, `settings.json`, and `experiment_metadata.json`: method versions, parameters, paths, and timing protocols;
- `*.png`, `*.pdf`, and `*.svg`: manuscript or diagnostic figures generated by the corresponding plotting scripts.

## Result-management conventions

1. Do not overwrite the original-manuscript results in `results/ds_proj_paper/`.
2. Write all revision experiments to `results/ds_proj_revise_V1/`.
3. Preserve `comparison/common_conditions/`, which contains the shared test conditions required for fair cross-method comparisons.
4. Store the formal raw feasibility and projection-optimality errors as squared distances. The relative-percentage presentation in the uncertainty experiment follows its own documented convention.
5. Plotting scripts read formal results only and do not repeat time-consuming AC-physics training.
6. Before deleting or moving results, verify that each summary, raw array, model weight, and metadata set remains complete.
7. Completed-state checkpoints, per-step training frames, pilot outputs, and migration backups are local artifacts and are not included in the public repository.

## Runtime environment

Create the core environment from the repository root:

```powershell
conda env create -f environment.yml
conda activate pinn-tdi-fr
```

The recorded experiments used Python 3.13.5, Ipopt 3.14.19, and the development CUDA build `torch 2.12.0.dev20260411+cu130`. Because development PyTorch wheels are not guaranteed to remain available indefinitely, `environment.yml` installs the current compatible PyTorch package instead of claiming that it can reproduce an archived nightly wheel. Install a CUDA-enabled PyTorch build appropriate for the local driver when GPU execution is required.

Gurobi requires a separate valid license. Ipopt must be discoverable as the `ipopt` executable. Before running a time-consuming formal experiment, use the script's `--help` or `--check-only` option, when available, to verify its parameters, weights, solvers, and CUDA environment. The committed numerical results and model weights can be inspected without retraining.
