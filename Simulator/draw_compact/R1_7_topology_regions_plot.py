# -*- coding: utf-8 -*-
"""Compare nominal Base/PINN/reconfigured regions in one PDF.

Python + NumPy + SciPy + Matplotlib. Physical regions are finite-direction
support envelopes reconstructed from saved minimizers, not exact AC sets.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle, ConnectionPatch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / 'results/ds_proj_revise_V1/comparison/topology/case33bw_ds'
DEFAULT_OUT = ROOT / 'results/pictures_compact/topology'
MAIN_LIMITS = ((0.25, 0.50), (0.16, 0.32))
TOPOLOGIES = ('Base', 'T1', 'T2', 'T3', 'T4', 'T5')
BASE, PINN, CHANGED, OUTSIDE = '#777777', '#315D88', '#318A80', '#BB5355'
STYLE = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10, 'axes.labelsize': 10,
    'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'pdf.fonttype': 42, 'hatch.linewidth': 0.5, 'savefig.facecolor': 'white',
}


def area(poly):
    if len(poly) < 3:
        return 0.0
    return abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                            - poly[:, 1] * np.roll(poly[:, 0], -1)))) / 2


def hull(points):
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if not np.isfinite(points).all() or len(points) < 3:
        raise ValueError('Incomplete or degenerate boundary points.')
    return points[ConvexHull(points).vertices]


def pinn_vertices(A, b):
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    norms = np.linalg.norm(A, axis=1)
    if np.any(norms <= 0):
        raise ValueError('Degenerate PINN inequality.')
    A, b = A / norms[:, None], b / norms
    points = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            matrix = A[[i, j]]
            if abs(np.linalg.det(matrix)) < 1e-11:
                continue
            point = np.linalg.solve(matrix, b[[i, j]])
            if np.all(A @ point <= b + 1e-9):
                points.append(point)
    poly = hull(points)
    if np.max(A @ poly.T - b[:, None]) > 1e-8:
        raise ValueError('Reconstructed PINN vertices violate constraints.')
    return poly


def clip(poly, normal, offset, keep_inside=True):
    """Clip a convex polygon by one half-plane without raster approximations."""
    if len(poly) == 0:
        return np.empty((0, 2))
    distances = poly @ normal - offset
    if not keep_inside:
        distances = -distances
    output = []
    for i, end in enumerate(poly):
        start = poly[i - 1]
        ds, de = distances[i - 1], distances[i]
        sin, ein = ds <= 0, de <= 0
        if sin != ein:
            output.append(start + ds / (ds - de) * (end - start))
        if ein:
            output.append(end)
    return np.asarray(output).reshape(-1, 2)


def difference(subject, boundary):
    """Partition subject minus a CCW convex boundary into disjoint polygons."""
    remaining = subject.copy()
    pieces = []
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0)):
        edge = end - start
        normal = np.array([edge[1], -edge[0]])
        offset = normal @ start
        piece = clip(remaining, normal, offset, keep_inside=False)
        if area(piece) > 1e-14:
            pieces.append(piece)
        remaining = clip(remaining, normal, offset)
    np.testing.assert_allclose(area(remaining) + sum(map(area, pieces)),
                               area(subject), rtol=1e-8, atol=1e-12)
    return pieces, remaining


def load_regions(data_root, sample):
    regions, metadata = {}, {}
    A_base = b_base = theta = directions = None
    for name in TOPOLOGIES:
        with np.load(data_root / name / 'evaluation_data.npz', allow_pickle=False) as data:
            points = data['true_support'][sample]
            A, b = data['predicted_A'][sample], data['predicted_b'][sample]
            if name == 'Base':
                A_base, b_base = A.copy(), b.copy()
                theta = data['dthetas'][sample].copy()
                directions = data['eval_dirs'].copy()
            else:
                np.testing.assert_allclose(A, A_base)
                np.testing.assert_allclose(b, b_base)
                np.testing.assert_allclose(data['dthetas'][sample], theta)
                np.testing.assert_allclose(data['eval_dirs'], directions)
            if not np.isfinite(points).all():
                raise ValueError('Incomplete saved support evaluations.')
            # optimize_direction minimizes d.x: the supporting half-plane is
            # d.x >= d.x_support. Chords between support points would create
            # spurious outside slivers even for feasible PINN vertices.
            normals = -data['eval_dirs']
            offsets = np.einsum('ij,ij->i', normals, points)
            regions[name] = pinn_vertices(normals, offsets)
            metadata[name] = {
                'closed': data['closed_tie'].tolist(),
                'opened': data['opened_line'].tolist(),
            }
    metadata['sample'] = sample
    metadata['n_directions'] = len(directions)
    metadata['nominal'] = bool(np.allclose(theta, 0))
    return regions, pinn_vertices(A_base, b_base), metadata


def outline(ax, poly, **kwargs):
    ring = np.vstack((poly, poly[0]))
    ax.plot(ring[:, 0], ring[:, 1], **kwargs)


def layers(ax, base, pinn, changed, pieces):
    ax.add_patch(Polygon(base, facecolor='#F0F0F0', edgecolor='none', zorder=1))
    ax.add_patch(Polygon(pinn, facecolor='#DCE8F2', edgecolor='none', zorder=2))
    for piece in pieces:
        ax.add_patch(Polygon(piece, facecolor='#F0BDB8', edgecolor=OUTSIDE,
                             linewidth=0, hatch='////', zorder=3))
    outline(ax, base, color=BASE, linestyle=(0, (4, 2)), linewidth=1.0, zorder=4)
    for boundary in changed:
        outline(ax, boundary, color=CHANGED, linewidth=0.85, alpha=0.8, zorder=5)
    outline(ax, pinn, color=PINN, linewidth=1.15, zorder=6)


def zoom_center(pieces, boundary):
    """Focus on the outside vertex furthest from the sampled physical boundary."""
    vertices = np.concatenate(pieces)
    starts = boundary
    edges = np.roll(boundary, -1, axis=0) - starts
    delta = vertices[:, None, :] - starts[None, :, :]
    t = np.clip(np.sum(delta * edges, axis=-1) / np.sum(edges**2, axis=-1), 0, 1)
    projections = starts[None, :, :] + t[..., None] * edges
    distances = np.linalg.norm(vertices[:, None, :] - projections, axis=-1).min(axis=1)
    return vertices[np.argmax(distances)]


def make_figure(regions, pinn, limits=MAIN_LIMITS):
    base = regions['Base']
    changed = [regions[name] for name in TOPOLOGIES[1:]]
    common = changed[0].copy()
    for boundary in changed[1:]:
        _, common = difference(common, boundary)
    # Union of exceedances: P minus the intersection of changed envelopes.
    pieces, intersection = difference(pinn, common)
    fig = plt.figure(figsize=(183 / 25.4, 90 / 25.4))
    ax = fig.add_axes([0.085, 0.27, 0.535, 0.696])
    layers(ax, base, pinn, changed, pieces)
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Active power at TDI (p.u.)')
    ax.set_ylabel('Reactive power at TDI (p.u.)')
    ax.set_xticks(np.linspace(0.25, 0.50, 6))
    ax.set_yticks(np.linspace(0.16, 0.32, 5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(False)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=BASE, linestyle='--', linewidth=1,
                      label='Region boundary of the base topology'),
               Line2D([], [], color=CHANGED, linewidth=0.85,
                      label='Region boundaries of changed topologies'),
               Patch(facecolor='#DCE8F2', edgecolor=PINN, label='PINN region'),
               Patch(facecolor='#F0BDB8', edgecolor=OUTSIDE, hatch='////',
                     label='PINN exceedance')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.535, 0.01),
               ncol=2, frameon=False, fontsize=9, handlelength=2.3,
               columnspacing=1.8, labelspacing=0.7)

    # The zoom occupies a separate axes to the right of the main plot.
    if sum(map(area, pieces)) > 1e-10:
        center = zoom_center(pieces, common)
        span = 0.035
        step = 0.01
        zoom_low = np.floor((center - span / 2) / step) * step
        zoom_high = np.ceil((center + span / 2) / step) * step
        outside_points = np.concatenate(pieces)
        zoom_low = np.minimum(zoom_low, np.floor(outside_points.min(axis=0) / step) * step)
        zoom_high = np.maximum(zoom_high, np.ceil(outside_points.max(axis=0) / step) * step)
        inset = fig.add_axes([0.67, 0.355, 0.27, 0.55])
        inset.set_facecolor('white')
        inset.grid(False)
        layers(inset, base, pinn, changed, pieces)
        inset.set_xlim(zoom_low[0], zoom_high[0])
        inset.set_ylim(zoom_low[1], zoom_high[1])
        inset.set_aspect('equal')
        inset.set_xticks(np.linspace(zoom_low[0], zoom_high[0],
                                     round((zoom_high[0] - zoom_low[0]) / step) + 1))
        inset.set_yticks(np.linspace(zoom_low[1], zoom_high[1],
                                     round((zoom_high[1] - zoom_low[1]) / step) + 1))
        inset.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        inset.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        inset.tick_params(labelsize=8.5, length=2.5, pad=2)
        inset.yaxis.tick_right()
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color('#999999')
            spine.set_linewidth(0.55)
        ax.add_patch(Rectangle(zoom_low, *(zoom_high - zoom_low),
                               fill=False, edgecolor='#999999', linewidth=0.65, zorder=7))
        for source_y, target_y in ((zoom_low[1], 0), (zoom_high[1], 1)):
            connector = ConnectionPatch(
                xyA=(zoom_high[0], source_y), coordsA=ax.transData,
                xyB=(0, target_y), coordsB=inset.transAxes,
                color='#AAAAAA', linewidth=0.65, clip_on=False, zorder=0)
            fig.add_artist(connector)
    return fig, pieces


def write_notes(out, metadata):
    condition = 'Nominal operating conditions (dtheta=0) ' if metadata['nominal'] else f"No.{metadata['sample']}No. operating condition"
    lines = [
        '1. Implementation details of the paper',
        f"adopt33node system{condition}, Fixed basic topology trainingPINN, No retraining; compareBaseandT1–T5Reconstruct the topology. use{metadata['n_directions']}The saved minimum support values in each direction construct a half-plane intersection, which represents the finite direction support envelope of the true feasible region and avoids mislabeling gaps caused by chords between support points as exceeding.PINNThe area is reconstructed by its linear inequality, and the red area is calculated by the polygon difference set; not marked in red does not mean that there is no cross-border, and small cross-borders may not be resolved by the sampling direction..",
        '',
        'Two, caption',
        'The gray dashed line and light gray filling represent the basic topological support envelope, and the blue border and light blue filling represent fixedPINNArea, all green boundaries uniformly represent changing topology, without distinguishing specific types. Red slash indicatesPINNThe part that exceeds at least one of the changing topological envelopes, that is, the union of the excess areas; the enlarged image on the right shows the local excess area.',
        '',
        'Topological operations (close tie line; disconnect original branch) : ',
    ]
    for name in TOPOLOGIES[1:]:
        lines.append(f"{name}: {metadata[name]['closed']}; {metadata[name]['opened']}")
    (out / 'implementation_notes.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--sample', type=int, default=0)
    args = parser.parse_args()
    if args.sample < 0:
        parser.error('--sample must be nonnegative')
    regions, pinn, metadata = load_regions(args.data_root, args.sample)
    args.out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig, pieces = make_figure(regions, pinn)
        fig.savefig(args.out / 'topology_regions_comparison.pdf', format='pdf')
        plt.close(fig)
        print('Outside at least one changed envelope: '
              f'{sum(map(area, pieces)) / area(pinn) * 100:.6f}% of PINN area')
    write_notes(args.out, metadata)
    print('Saved one combined PDF figure and concise notes.')


if __name__ == '__main__':
    main()
