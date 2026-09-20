# -*- coding: utf-8 -*-
"""Plot the R2.4 measurement-uncertainty error distributions."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from Simulator import PROJECT_ROOT


CASENAME = 'case33bw_ds'
DEFAULT_DIR = (
    PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison'
    / 'uncertainty' / CASENAME
)
PLOT_FLOOR = 1e-12


def valid_values(errors, success):
    """Flatten and exclude only solution failure records; true zero errors remain."""
    return np.asarray(errors)[np.asarray(success, dtype=bool)]


def styled_boxplot(ax, datasets, labels, color, title, ylabel):
    display = [np.maximum(x, PLOT_FLOOR) for x in datasets]
    bp = ax.boxplot(
        display,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.58,
        whis=(5, 95),
    )
    for box in bp['boxes']:
        box.set(facecolor=color, edgecolor=color, alpha=0.32, linewidth=1.3)
    for median in bp['medians']:
        median.set(color=color, linewidth=2.0)
    for key in ('whiskers', 'caps'):
        for artist in bp[key]:
            artist.set(color=color, linewidth=1.1)
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel('Measurement-error standard deviation')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.28)


def signed_boxplot(ax, datasets, labels, color, title, ylabel):
    """Plots the pairing error increment that is allowed to be negative; the zero line represents the same as the noise-free baseline."""
    bp = ax.boxplot(
        datasets, tick_labels=labels, patch_artist=True,
        showfliers=False, widths=0.58, whis=(5, 95),
    )
    for box in bp['boxes']:
        box.set(facecolor=color, edgecolor=color, alpha=0.32, linewidth=1.3)
    for median in bp['medians']:
        median.set(color=color, linewidth=2.0)
    for key in ('whiskers', 'caps'):
        for artist in bp[key]:
            artist.set(color=color, linewidth=1.1)
    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--')
    ax.set_yscale('symlog', linthresh=1e-10)
    ax.set_title(title)
    ax.set_xlabel('Measurement-error standard deviation')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.28)


def subset_means(errors, success, out_of_range):
    """Calculate the training range separately/Mean direction error of external measurement samples."""
    means = {'in_range': [], 'out_of_range': []}
    for i in range(errors.shape[0]):
        for name, sample_mask in (
            ('in_range', ~out_of_range[i]),
            ('out_of_range', out_of_range[i]),
        ):
            mask = success[i] & np.broadcast_to(
                sample_mask[..., None], errors[i].shape)
            values = errors[i][mask]
            means[name].append(float(values.mean()) if values.size else np.nan)
    return means


def main():
    parser = argparse.ArgumentParser(
        description='R2.4 case33bw_ds Measurement uncertainty error distribution chart')
    parser.add_argument('--data', type=Path,
                        default=DEFAULT_DIR / 'uncertainty_data.npz')
    parser.add_argument('--out', type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f'Data not found: {args.data}\n'
            'Please run first main_revise2_4_uncertainty.py.')

    data = np.load(args.data)
    casename = str(data['casename'])
    noise_levels = data['noise_levels']
    feas = data['feasibility_error_percentages']
    opt = data['projection_optimality_error_percentages']
    feas_success = data['feasibility_solve_success']
    opt_success = data['optimality_solve_success']
    feas_inc = data['feasibility_percentage_increments']
    opt_inc = data['projection_optimality_percentage_increments']
    paired_feas_success = data['paired_feasibility_solve_success']
    paired_opt_success = data['paired_projection_optimality_solve_success']
    out_of_range = data['out_of_range_measurement_mask']

    if feas.shape[0] != len(noise_levels) or opt.shape != feas.shape:
        raise ValueError('Error array dimensions and noise_levels inconsistent.')

    labels = [f'{100*x:g}%' for x in noise_levels]
    feas_dist = [valid_values(feas[i], feas_success[i])
                 for i in range(len(noise_levels))]
    opt_dist = [valid_values(opt[i], opt_success[i])
                for i in range(len(noise_levels))]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    styled_boxplot(
        axes[0], feas_dist, labels, '#C44E52',
        '(a) Feasibility-error distribution',
        'Feasibility boundary error (% of root reference power)',
    )
    styled_boxplot(
        axes[1], opt_dist, labels, '#4C72B0',
        '(b) Projection-based optimality-error distribution',
        'Projection-based optimality error (% of root reference power)',
    )

    x = np.arange(len(noise_levels))
    oor_rate = out_of_range.mean(axis=(1, 2)) * 100.0
    axes[2].plot(x, oor_rate, marker='o', linewidth=2.0,
                 color='#8172B2', label='At least one input out of range')
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 105)
    axes[2].set_xlabel('Measurement-error standard deviation')
    axes[2].set_ylabel('Rate (%)')
    axes[2].set_title('(c) Out-of-training-range measurement rate')
    axes[2].grid(linestyle='--', alpha=0.28)
    axes[2].legend(fontsize=8)

    n_true, n_repeat, n_dirs = feas.shape[1:]
    fig.suptitle(
        f'Measurement uncertainty — {casename} '
        f'({n_true} true cases × {n_repeat} repeats × {n_dirs} directions)',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / 'uncertainty_error_distribution.png'
    svg = args.out / 'uncertainty_error_distribution.svg'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)

    # Pairing increment: each sigma With the same operating condition, same repetition, and same direction 0% base subtraction.
    feas_inc_dist = [valid_values(feas_inc[i], paired_feas_success[i])
                     for i in range(len(noise_levels))]
    opt_inc_dist = [valid_values(opt_inc[i], paired_opt_success[i])
                    for i in range(len(noise_levels))]
    fig2, axes2 = plt.subplots(1, 2, figsize=(10.6, 4.5))
    signed_boxplot(
        axes2[0], feas_inc_dist, labels, '#C44E52',
        '(a) Paired feasibility-error increment',
        r'$E_{feas}(\sigma)-E_{feas}(0)$ (percentage points)',
    )
    signed_boxplot(
        axes2[1], opt_inc_dist, labels, '#4C72B0',
        '(b) Paired projection-optimality-error increment',
        r'$E_{opt}^{proj}(\sigma)-E_{opt}^{proj}(0)$ (percentage points)',
    )
    fig2.tight_layout()
    inc_png = args.out / 'uncertainty_paired_error_increment.png'
    inc_svg = args.out / 'uncertainty_paired_error_increment.svg'
    fig2.savefig(inc_png, dpi=300, bbox_inches='tight')
    fig2.savefig(inc_svg, bbox_inches='tight')
    plt.close(fig2)

    # Separate in-range perturbations from out-of-distribution extrapolation.
    feas_subset = subset_means(feas, feas_success, out_of_range)
    opt_subset = subset_means(opt, opt_success, out_of_range)
    fig3, axes3 = plt.subplots(1, 2, figsize=(10.6, 4.5))
    for ax, values, title, ylabel in (
        (axes3[0], feas_subset, '(a) Feasibility error by input range',
         'Mean feasibility error (% of root reference power)'),
        (axes3[1], opt_subset,
         '(b) Projection-based optimality error by input range',
         'Mean projection error (% of root reference power)'),
    ):
        in_values = np.asarray(values['in_range'], dtype=float)
        out_values = np.asarray(values['out_of_range'], dtype=float)
        in_display = np.where(
            np.isfinite(in_values), np.maximum(in_values, PLOT_FLOOR), np.nan)
        out_display = np.where(
            np.isfinite(out_values), np.maximum(out_values, PLOT_FLOOR), np.nan)
        ax.plot(x, in_display, marker='o', label='In training range')
        ax.plot(x, out_display, marker='s', label='Out of training range')
        ax.set_xticks(x, labels)
        ax.set_yscale('log')
        ax.set_xlabel('Measurement-error standard deviation')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(which='both', linestyle='--', alpha=0.28)
        ax.legend(fontsize=8)
    fig3.tight_layout()
    subset_png = args.out / 'uncertainty_in_range_vs_ood.png'
    subset_svg = args.out / 'uncertainty_in_range_vs_ood.svg'
    fig3.savefig(subset_png, dpi=300, bbox_inches='tight')
    fig3.savefig(subset_svg, bbox_inches='tight')
    plt.close(fig3)

    print(f'saved: {png}')
    print(f'saved: {svg}')
    print(f'saved: {inc_png}')
    print(f'saved: {inc_svg}')
    print(f'saved: {subset_png}')
    print(f'saved: {subset_svg}')


if __name__ == '__main__':
    main()
