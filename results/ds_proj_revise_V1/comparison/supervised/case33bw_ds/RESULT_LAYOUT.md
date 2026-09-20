# R1.6 Supervised Result Layout

- `support_points/labels/`: formal support-point (vertex-method) labels. Each operating condition contains AC feasible-region support-optimization points in 36 fixed unit directions.
- `support_points/models/n{N}/`: supervised networks, checkpoints, and training metadata for label-set sizes `N=50/100/200/500/1000/2000/5000`.
- `support_points/evaluation/`: raw directional evaluations of the PINN and seven supervised networks on the same 50 test conditions.
- `support_points/summary.csv`: summary of offline cost, boundary-point MSE, squared error, and coverage.
- `models/pinn_retrained/`: the R1.6 PINN model used in the comparison with the supervised baselines.

The sole source of formal online times is `comparison/learning_methods/case33bw_ds/online_manuscript_protocol_current_hardware/online_manuscript_protocol_times.csv`. Timing uses independent subprocesses, excludes model loading and `PreTrainNet` initialization, does not warm up the target network, and measures one first complete region output.
