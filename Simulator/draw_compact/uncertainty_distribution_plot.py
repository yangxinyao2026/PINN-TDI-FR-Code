# -*- coding: utf-8 -*-
"""Publication figure for measurement-noise effects on root-normalized errors.

Run directly with Python; only NumPy and Matplotlib are required. Original
squared errors are converted to 100*sqrt(e/(P_ref^2+Q_ref^2)) exactly once.
Two PDF panels and a TXT note are written to pictures_compact/uncertainty.
Boxes: P25-P75, median, 1.5-IQR whiskers; scatter: all successful evaluations.
Lines connect group maxima. Directions and noise realizations are nested
within operating conditions and paired across noise levels; no significance test.
Feasibility uses a log scale (symlog when zeros occur); optimality is linear.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np


DEFAULT_DIR = (Path(__file__).resolve().parents[2] / 'results'
               / 'ds_proj_revise_V1' / 'comparison' / 'uncertainty'
               / 'case33bw_ds')
DEFAULT_OUT = (Path(__file__).resolve().parents[2] / 'results'
               / 'pictures_compact' / 'uncertainty')
STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'axes.linewidth': 0.65,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.65,
    'ytick.major.width': 0.65,
    'pdf.fonttype': 42,
    'legend.frameon': False,
    'savefig.facecolor': 'white',
}
METRICS = (
    ('feasibility', 'feasibility_errors', 'feasibility_solve_success',
     'feasibility_error_percentages', 'Feasibility', '#B65359', '#E8A8AA'),
    ('optimality', 'projection_optimality_errors',
     'projection_optimality_solve_success',
     'projection_optimality_error_percentages',
     'Projection-based optimality', '#386F96', '#A7CCDF'),
)


def load_distributions(path):
    """Recompute the plotted quantity and cross-check saved percentages."""
    with np.load(path, allow_pickle=False) as data:
        levels = np.asarray(data['noise_levels'], dtype=float)
        denom = np.asarray(data['reference_power_pu_squared'], dtype=float)
        if (levels.ndim != 1 or len(levels) < 2
                or not np.all(np.isfinite(levels)) or np.any(levels < 0)
                or np.any(np.diff(levels) <= 0)):
            raise ValueError('Noise levels must be finite, nonnegative and increasing.')
        if not np.all(np.isfinite(denom)) or np.any(denom <= 0):
            raise ValueError('Reference squared power must be finite and positive.')
        groups, masks = [], []
        shape = None
        for _, raw_key, mask_key, pct_key, *_ in METRICS:
            raw = np.asarray(data[raw_key], dtype=float)
            success = np.asarray(data[mask_key], dtype=bool)
            if (raw.ndim != 4 or raw.shape[0] != len(levels)
                    or denom.shape != (raw.shape[1],) or success.shape != raw.shape
                    or (shape is not None and raw.shape != shape)):
                raise ValueError('Inconsistent noise / condition / repeat / direction axes.')
            shape = raw.shape
            if np.any(~np.isfinite(raw[success])) or np.any(raw[success] < 0):
                raise ValueError('Successful solves contain invalid squared errors.')
            percentages = 100 * np.sqrt(np.where(success, raw, np.nan)
                                        / denom[None, :, None, None])
            if not np.allclose(percentages[success], data[pct_key][success],
                               rtol=1e-10, atol=1e-12):
                raise ValueError('Saved percentages disagree with the square-root definition.')
            if any(not np.any(mask) for mask in success):
                raise ValueError('A noise level has no successful evaluations.')
            groups.append(percentages)
            masks.append(success)
        return levels, groups, masks


def align_y_limits_to_major_ticks(ax):
    lower, upper = ax.get_ylim()
    if ax.get_yscale() == 'log':
        lower, upper = ax.dataLim.intervaly
        ticks = 10.0 ** np.arange(np.floor(np.log10(lower)),
                                  np.ceil(np.log10(upper)) + 1)
    elif ax.get_yscale() == 'symlog':
        threshold = ax.yaxis.get_transform().linthresh
        ticks = np.r_[0.0, 10.0 ** np.arange(np.floor(np.log10(threshold)),
                                            np.ceil(np.log10(upper)) + 1)]
    else:
        ticks = ax.yaxis.get_major_locator().tick_values(lower, upper)
        start = np.flatnonzero(ticks <= lower)[-1]
        stop = np.flatnonzero(ticks >= upper)[0]
        ticks = ticks[start:stop + 1]
    ax.set_ylim(ticks[0], ticks[-1])
    ax.yaxis.set_major_locator(FixedLocator(ticks))


def draw_panel(ax, levels, values, success, metric):
    name, _, _, _, _, edge, fill = metric
    x = levels * 100
    width = min(np.diff(x)) * 0.55
    distributions = [v[m] for v, m in zip(values, success)]
    # Preserve true zeros if present.
    if name == 'feasibility':
        if any(np.any(v == 0) for v in distributions):
            positives = np.concatenate([v[v > 0] for v in distributions])
            threshold = max(float(positives.min()), 1e-12) if positives.size else 1e-12
            ax.set_yscale('symlog', linthresh=threshold)
        else:
            ax.set_yscale('log')
    rng = np.random.default_rng(42)
    for xi, vals in zip(x, distributions):
        ax.scatter(xi + rng.uniform(-width * 0.36, width * 0.36, vals.size),
                   vals, s=2.8, c=fill, alpha=0.28, edgecolors='none',
                   zorder=1, rasterized=True)
    ax.boxplot(
        distributions, positions=x, widths=width, manage_ticks=False,
        patch_artist=True, showfliers=False, whis=1.5,
        boxprops={'facecolor': fill, 'edgecolor': edge, 'linewidth': 0.8,
                  'alpha': 0.65},
        medianprops={'color': edge, 'linewidth': 1.3},
        whiskerprops={'color': edge, 'linewidth': 0.8},
        capprops={'color': edge, 'linewidth': 0.8}, zorder=2,
    )
    maxima = np.array([np.max(v) for v in distributions])
    ax.plot(x, maxima, '-o', color='#292929', markersize=3.3,
            markerfacecolor='white', markeredgewidth=0.8, linewidth=1, zorder=4)
    ax.set_xticks(x, [f'{v:g}' for v in x])
    ax.set_xlim(x[0] - width * 1.3, x[-1] + width * 1.3)
    if name == 'optimality':
        ax.set_ylim(bottom=0)
    ax.margins(y=0.09)
    align_y_limits_to_major_ticks(ax)
    ax.set_xlabel('Standard deviation of measurement noise (%)')
    ax.set_ylabel(f'Square-root {name} error rate (%)')
    ax.grid(axis='y', which='major', color='#D8D8D8', linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)


def export_figure(fig, out, stem):
    # Fixed canvas preserves physical publication dimensions; PDF only.
    fig.savefig(out / f'{stem}.pdf', format='pdf', dpi=600)


def write_notes(out, data_path, levels, groups, masks):
    n_true, n_repeat, n_dirs = groups[0].shape[1:]
    with np.load(data_path, allow_pickle=False) as data:
        case = str(data['casename'].item())
        seed = int(data['seed'])
    noise_text = ', '.join(f'{100*v:g}%' for v in levels)
    notes = [
        '1. Implementation details that need to be explained in the paper',
        f'adopt {case} The model trained by the calculation example and manuscript applies independent multiplicative Gaussian noise to the active and reactive load measurements, and the standard deviation is {noise_text}.The network uses noisy input without clipping, and the true feasible region is determined by noise-free parameters..',
        f'Test contains {n_true} real operating conditions, each operating condition {n_repeat} Noise realization, every time {n_dirs} directions; each noise level reuses operating conditions, standard noise and directions, and the random seed is {seed}.',
        'The error ratio is 100√[e/(P_ref²+Q_ref²)]%, Among them e is the square projection distance, and the reference power is taken from the root node operating point of each real operating condition; optimality refers to the geometric projection error.',
        '',
        '2. Help in reading pictures caption content',
        f'Root feasibility and optimality error ratio distribution under different measurement noise levels. Each noise level contains {n_true*n_repeat*n_dirs} Observations are made in several directions, and only records with successful solutions are drawn. The box represents the25–75Percentile, the horizontal line inside the box represents the median, and the whiskers extend to1.5The farthest observation within twice the interquartile range, the scattered points represent each observation; the black open circle line connects the maximum observation value of each group. The feasibility diagram uses a logarithmic vertical axis, and the optimality diagram uses a linear vertical axis..',
    ]
    (out / 'implementation_notes.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DIR / 'uncertainty_data.npz')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    levels, groups, masks = load_distributions(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for group, mask, metric in zip(groups, masks, METRICS):
            fig, ax = plt.subplots(figsize=(95 / 25.4, 80 / 25.4))
            fig.subplots_adjust(left=0.19, right=0.975, top=0.975, bottom=0.16)
            draw_panel(ax, levels, group, mask, metric)
            export_figure(fig, args.out, f'uncertainty_sqrt_{metric[0]}_ratio')
            plt.close(fig)
    write_notes(args.out, args.data, levels, groups, masks)
    print('Exported two PDF panels and implementation_notes.txt.')
    print(f'Validated normalization; successful observations: {[int(m.sum()) for m in masks]}')


if __name__ == '__main__':
    main()
