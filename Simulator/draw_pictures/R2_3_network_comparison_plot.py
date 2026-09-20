# -*- coding: utf-8 -*-
"""Plot the R2.3 neural-network architecture comparison."""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from Simulator import PROJECT_ROOT

CASENAME = 'case33bw_ds'
OUT = PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' / 'comparison' / 'network' / CASENAME
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
FORMAL_CONFIGS = [
    'h128_act-relu', 'h128_act-tanh', 'h128_act-silu',
    'h128-128_act-relu', 'h128-128_act-tanh', 'h128-128_act-silu',
]


def load_summary():
    path = OUT / 'formal_summary.npz'
    if not path.exists():
        raise FileNotFoundError(
            f"not found {path}, Please run first main_revise2_3_network.py --stage evaluate")
    d = np.load(path, allow_pickle=True)
    summary = {k: d[k] for k in d.files}
    keep = np.array([str(tag) in FORMAL_CONFIGS for tag in summary['config']])
    if not np.any(keep):
        raise ValueError('summary.npz None R2.3 Formal six configuration results.')
    return {k: v[keep] for k, v in summary.items()}


def index_metric(s, metric):
    """Press the one-dimensional indicator array by (hidden_sizes, activation) The index is {hidden: {act: value}}."""
    out = {}
    for h, a, v in zip(s['hidden_sizes'], s['activation'], s[metric]):
        out.setdefault(str(h), {})[str(a)] = float(v)
    return out


def grouped_bars(ax, s, metric, ylabel, title):
    """xThe axis is ReLU/Tanh/SiLU, Each hidden layer structure is represented by a set of bars."""
    idx = index_metric(s, metric)
    hidden_list = list(idx.keys())                 # ['128', '128,128']
    acts, seen = [], set()                          # Take the union of all activations that have occurred, keeping the order of first appearance.
    for h in hidden_list:
        for a in idx[h].keys():
            if a not in seen:
                acts.append(a); seen.add(a)
    x = np.arange(len(acts))
    n = len(hidden_list)
    w = 0.8 / n
    for i, h in enumerate(hidden_list):
        vals = [idx[h].get(a, np.nan) for a in acts]   # Untested activation settings nan, The corresponding column is not displayed
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w,
                      label=f'hidden [{h}]', color=COLORS[i % len(COLORS)])
        for r, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f'{v:.2e}',
                            (r.get_x() + r.get_width() / 2, v),
                            ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(acts, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)


def main():
    s = load_summary()

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(f'R2.3 Network structure × activation comparison  ({CASENAME})')

    grouped_bars(axes[0, 0], s, 'feas_mean',  'feasibility error (mean)', '(a) Feasibility error')
    grouped_bars(axes[0, 1], s, 'opt_mean',   'optimality error (mean)',  '(b) Optimality error')
    grouped_bars(axes[0, 2], s, 'coverage_mean', r'coverage $\rho$ (mean)',
                 r'(c) Reference-point coverage $\rho=k_P/k_\Omega$')
    axes[0, 2].axhline(1.0, color='#C44E52', ls='--', lw=1.0)
    grouped_bars(axes[1, 0], s, 'parameter_count', 'trainable parameters',
                 '(d) Model size')

    # (e) Box plot of error distribution for each configuration: directory name is taken directly from summary of config Field
    ax = axes[1, 1]
    tags = [str(t) for t in s['config']]           # Such as 'h128_act-relu','h128-128_act-tanh'
    feas_boxes, opt_boxes, positions = [], [], []
    for i, t in enumerate(tags):
        ep = OUT / t / 'formal_eval.npz'
        if not ep.exists():
            continue
        e = np.load(ep)
        feas_boxes.append(e['feasibility'][e['feasibility_success']])
        opt_boxes.append(e['optimality'][e['optimality_success']])
        positions.append(i)
    if feas_boxes:
        bp1 = ax.boxplot(feas_boxes, positions=[p - 0.18 for p in positions],
                         widths=0.32, showfliers=False, patch_artist=True)
        bp2 = ax.boxplot(opt_boxes, positions=[p + 0.18 for p in positions],
                         widths=0.32, showfliers=False, patch_artist=True)
        for b in bp1['boxes']:
            b.set(facecolor='#4C72B0', alpha=0.6)
        for b in bp2['boxes']:
            b.set(facecolor='#DD8452', alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([t.replace('h', '').replace('_act-', '\n') for t in tags],
                           fontsize=6)
        ax.set_ylabel('error distribution')
        ax.set_title('(e) Error distribution  (blue=feas, orange=opt)')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_yscale('log')

    # (f) The original distribution of the coverage of each configuration reference point; the solution failed NaN Do not enter box plot.
    ax = axes[1, 2]
    coverage_boxes, coverage_positions = [], []
    for i, t in enumerate(tags):
        ep = OUT / t / 'formal_eval.npz'
        if not ep.exists():
            continue
        e = np.load(ep)
        if 'coverage' not in e.files:
            continue
        values = np.asarray(e['coverage'], dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            coverage_boxes.append(values)
            coverage_positions.append(i)
    if coverage_boxes:
        bp = ax.boxplot(coverage_boxes, positions=coverage_positions,
                        widths=0.55, showfliers=False, patch_artist=True)
        for b in bp['boxes']:
            b.set(facecolor='#55A868', alpha=0.65)
        ax.axhline(1.0, color='#C44E52', ls='--', lw=1.0,
                   label=r'ideal $\rho=1$')
        ax.set_xticks(coverage_positions)
        ax.set_xticklabels(
            [tags[i].replace('h', '').replace('_act-', '\n')
             for i in coverage_positions], fontsize=6)
        ax.set_ylabel(r'coverage $\rho=k_P/k_\Omega$')
        ax.set_title('(f) Coverage distribution')
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = OUT.parent / 'network_comparison.png'
    out_svg = OUT.parent / 'network_comparison.svg'
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_svg)
    print(f"saved: {out_png}\n        {out_svg}")


if __name__ == '__main__':
    main()
