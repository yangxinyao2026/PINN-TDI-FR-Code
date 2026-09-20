# -*- coding: utf-8 -*-
"""Plot the R1.2 online-timing and break-even results."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Compatible fromPyCharm/Run this file directly from the command line.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Simulator import PROJECT_ROOT


ROOT = (PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' /
        'comparison' / 'amortization')
SUMMARY = ROOT / 'summary.csv'
ORIGINAL_ONLINE = ROOT / 'original_online_times.csv'


def read_rows():
    if not SUMMARY.exists():
        raise FileNotFoundError(
            f'not found {SUMMARY}, Please run first main_revise1_2_amortization.py')
    with open(SUMMARY, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def values(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=float)


def main():
    rows = read_rows()
    labels = [row['case'].replace('_ds', '').replace('case', '') for row in rows]
    x = np.arange(len(rows))
    width = 0.18

    if not ORIGINAL_ONLINE.exists():
        raise FileNotFoundError(
            f'not found {ORIGINAL_ONLINE}, Please rerunR1.2 runner')
    with open(ORIGINAL_ONLINE, 'r', encoding='utf-8') as f:
        online_rows = list(csv.DictReader(f))
    indexed = {
        (row['case'], int(row['m'])): row for row in online_rows}
    opt8 = np.asarray([
        float(indexed[(row['case'], 8)]['vertex_generation_s'])
        for row in rows])
    gpu8 = np.asarray([
        float(indexed[(row['case'], 8)]['pinn_gpu_s'])
        for row in rows])
    opt36 = np.asarray([
        float(indexed[(row['case'], 36)]['vertex_generation_s'])
        for row in rows])
    gpu36 = np.asarray([
        float(indexed[(row['case'], 36)]['pinn_gpu_s'])
        for row in rows])
    offline_h = values(rows, 'offline_training_h')
    be_gpu = values(rows, 'break_even_calls_gpu')
    days5_gpu = values(rows, 'days_5min_gpu')
    days15_gpu = values(rows, 'days_15min_gpu')

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))

    ax = axes[0, 0]
    ax.bar(x - 1.5 * width, opt8, width,
           label='Vertex, m=8', color='#9ecae1')
    ax.bar(x - 0.5 * width, gpu8, width,
           label='PINN, m=8', color='#a1d99b')
    ax.bar(x + 0.5 * width, opt36, width,
           label='Vertex, m=36', color='#4c78a8')
    ax.bar(x + 1.5 * width, gpu36, width,
           label='PINN, m=36', color='#59a14f')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Online computation time (s, log scale)')
    ax.set_title('(a) Original-manuscript online timing')
    ax.grid(axis='y', which='both', alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    bars = ax.bar(x, offline_h, color='#9c755f')
    ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Offline training time (h)')
    ax.set_title('(b) Pretraining + FullNet training')
    ax.grid(axis='y', alpha=0.25)

    ax = axes[1, 0]
    bars = ax.bar(x, be_gpu, width=0.52, color='#59a14f')
    ax.bar_label(bars, labels=[f'{int(value)}' for value in be_gpu],
                 fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of online region evaluations')
    ax.set_title('(c) Break-even point for GPU deployment')
    ax.grid(axis='y', alpha=0.25)

    ax = axes[1, 1]
    interval_width = 0.28
    bars5 = ax.bar(x - interval_width / 2, days5_gpu, interval_width,
                   label='5-min updates', color='#f2a541')
    bars15 = ax.bar(x + interval_width / 2, days15_gpu, interval_width,
                    label='15-min updates', color='#4c78a8')
    ax.bar_label(bars5, fmt='%.2f', fontsize=7, padding=2)
    ax.bar_label(bars15, fmt='%.2f', fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Days to break even')
    ax.set_title('(d) Amortization under different update intervals')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    png = ROOT / 'amortization_comparison.png'
    pdf = ROOT / 'amortization_comparison.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {png}')
    print(f'saved: {pdf}')


if __name__ == '__main__':
    main()
