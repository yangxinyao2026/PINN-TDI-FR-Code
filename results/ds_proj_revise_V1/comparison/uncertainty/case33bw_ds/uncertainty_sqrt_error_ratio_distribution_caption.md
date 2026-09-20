# Measurement uncertainty: square-root error-ratio distributions

The figure compares feasibility (a) and projection-based optimality (b) for case33bw_ds.
Each observation is 100 sqrt(e / (P_ref_pu^2 + Q_ref_pu^2)), where e is a squared
Euclidean projection distance and the denominator belongs to the true operating
condition. This is a distance ratio expressed as a percentage, not a second
square root of the saved percentage. Optimality means geometric projection error.

Boxes span P25–P75, internal lines are medians, and whiskers span P5–P95.
All successful observations, including tails, are shown as faint points; black
circles connect arithmetic means and dashed diamonds connect P95 values.
The noise axis uses actual numerical spacing. Feasibility uses a log axis
(symlog retaining zero if the input contains zeros); optimality uses a linear axis.
No observations are trimmed or winsorized. Scatter jitter uses a fixed seed.

There are 50 true operating conditions, 3 noise realizations per
condition and 10 directions per realization at each noise level. Conditions,
standard noise realizations and directions are paired across levels. Directional
observations are nested, not independent experimental replicates. Failed solves
are excluded and counted in the source summary. No significance test is used.
Simulation seed: 42.

The means and upper tails increase with noise in the supplied data. The feasibility
median stays near its baseline; the figure does not claim that all quantiles grow.
At high noise, the experiment includes inputs outside the training range.

Design: quantitative two-panel grid, robustness evidence; Times New Roman/STIX
retains the surrounding manuscript style, with restrained coral and blue fills.
Combined canvas: 183 × 90 mm. Individual panels: 95 × 86 mm.
Exports: editable-text SVG/PDF and 600 dpi PNG/TIFF. Scatter layers are rasterized.
Source rows preserve noise, condition, repeat and direction indices (zero-based).

Input SHA-256: 96ae5bcbab8da7ef16421d1dfbc08c8dbbd69997587ca45ce28457a224422383
