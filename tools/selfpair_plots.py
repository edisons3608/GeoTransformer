r"""Figures for the self-pair benchmark, built from the JSON files of `selfpair_eval.py`.

    python tools/selfpair_plots.py [output_dir]

Writes PNGs next to the JSON results (output/selfpair/figures by default).
"""

import glob
import json
import os
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', 'selfpair')

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e3e2de'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'text.color': INK,
    'axes.labelcolor': INK_2,
    'xtick.color': INK_2,
    'ytick.color': INK_2,
    'axes.edgecolor': GRID,
    'axes.linewidth': 1.0,
    'font.size': 9,
    'axes.titlesize': 10,
    'legend.frameon': False,
})


def load_results():
    runs = {}
    for path in sorted(glob.glob(osp.join(RESULT_DIR, '*.json'))):
        with open(path) as f:
            data = json.load(f)
        runs[data['summary']['tag']] = data
    return runs


def clean_axes(ax, grid_axis='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def style_box(bp, color):
    for box in bp['boxes']:
        box.set(facecolor=color + '33', edgecolor=color, linewidth=1.4)
    for part in ('whiskers', 'caps'):
        for item in bp[part]:
            item.set(color=color, linewidth=1.2)
    for median in bp['medians']:
        median.set(color=color, linewidth=2.0)
    for flier in bp.get('fliers', []):
        flier.set(marker='o', markersize=3, markerfacecolor='none', markeredgecolor=color, alpha=0.6)


def fig_sixdof(runs, out_path):
    r"""Signed per-axis 6-DoF error, rotation and translation in separate panels."""
    tags = [t for t in ('3dmatch_euler45', '3dmatch_so3180') if t in runs]
    if not tags:
        return None
    labels = {'3dmatch_euler45': 'rotations up to 45 degrees', '3dmatch_so3180': 'full random SO(3)'}
    colors = [BLUE, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = [
        ('rot_err_deg', ('rx', 'ry', 'rz'), 'rotation residual (degrees)', axes[0]),
        ('trans_err_mm', ('tx', 'ty', 'tz'), 'translation residual (mm)', axes[1]),
    ]
    for key, axis_names, ylabel, ax in panels:
        for run_idx, tag in enumerate(tags):
            errors = np.array([r[key] for r in runs[tag]['records']])
            positions = np.arange(3) + (run_idx - (len(tags) - 1) / 2) * 0.32
            bp = ax.boxplot([errors[:, i] for i in range(3)], positions=positions, widths=0.26,
                            patch_artist=True, showfliers=True)
            style_box(bp, colors[run_idx])
            bp['boxes'][0].set_label(labels.get(tag, tag))
        ax.axhline(0.0, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(axis_names)
        ax.set_ylabel(ylabel)
        clean_axes(ax)
    handles = [plt.Line2D([], [], color=c, linewidth=3) for c in colors]
    fig.legend(handles, [labels.get(t, t) for t in tags], loc='upper right',
               bbox_to_anchor=(0.99, 0.945), fontsize=8.5, ncol=2)
    fig.suptitle('GeoTransformer 3DMatch weights on talus self-pairs: per-axis 6-DoF error',
                 x=0.02, ha='left', fontsize=11, fontweight='bold')
    fig.text(0.02, 0.90, '93 pairs per configuration (31 tali x 3 trials); zero line = perfect alignment',
             fontsize=8.5, color=INK_2, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_run_comparison(runs, out_path):
    r"""Error magnitude per run: where each model lands, on a log scale."""
    order = sorted(runs, key=lambda t: np.median([r['rre_deg'] for r in runs[t]['records']]), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key, label in ((axes[0], 'rre_deg', 'rotation error RRE (degrees)'),
                           (axes[1], 'rmse_mm', 'point RMSE (mm)')):
        for i, tag in enumerate(order):
            values = np.array([r[key] for r in runs[tag]['records']])
            color = BLUE if runs[tag]['summary']['model'] == '3dmatch' else ORANGE
            bp = ax.boxplot([values], positions=[i], widths=0.55, orientation='horizontal', patch_artist=True, showfliers=True)
            style_box(bp, color)
            ax.text(np.median(values), i + 0.42, '{:.2f}'.format(np.median(values)),
                    fontsize=7.5, color=color, ha='center')
        ax.set_xscale('log')
        ax.set_xlabel(label)
        clean_axes(ax, grid_axis='x')
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order)
    handles = [plt.Line2D([], [], color=BLUE, linewidth=3), plt.Line2D([], [], color=ORANGE, linewidth=3)]
    axes[1].legend(handles, ['3DMatch weights', 'ModelNet weights'], loc='upper right', fontsize=8)
    fig.suptitle('Which pre-trained weights transfer to talus self-pairs', x=0.02, ha='left',
                 fontsize=11, fontweight='bold')
    fig.text(0.02, 0.90, 'log scale; box = quartiles, line = median (labelled), points = outliers',
             fontsize=8.5, color=INK_2, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_inlier_vs_error(runs, out_path):
    r"""Why the failures fail: correspondence quality against final rotation error."""
    groups = {}
    for tag in sorted(runs):
        model = runs[tag]['summary']['model']
        rotation = 'full random SO(3)' if 'so3' in tag else 'rotations up to 45 degrees'
        key = ('3DMatch' if model == '3dmatch' else 'ModelNet', rotation)
        groups.setdefault(key, []).extend(runs[tag]['records'])

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for (weights, rotation), records in sorted(groups.items()):
        color = BLUE if weights == '3DMatch' else ORANGE
        marker = '^' if rotation.startswith('full') else 'o'
        ax.scatter([r['inlier_ratio'] for r in records], [max(r['rre_deg'], 1e-2) for r in records],
                   s=22, facecolor='none', edgecolor=color, linewidth=1.0, marker=marker, alpha=0.7,
                   label='{} weights, {}'.format(weights, rotation))
    ax.axhline(5.0, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.text(0.015, 5.6, 'registration failure threshold (5 degrees)', fontsize=7.5, color=INK_2)
    ax.set_yscale('log')
    ax.set_xlabel('inlier ratio of predicted correspondences')
    ax.set_ylabel('rotation error RRE (degrees, log)')
    ax.set_title('Correspondence quality decides the registration', loc='left', fontweight='bold', pad=26)
    ax.text(0, 1.012, 'each point is one pair; the three ModelNet point-density runs are pooled',
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.legend(fontsize=8, loc='lower left')
    clean_axes(ax, grid_axis='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_per_specimen(runs, out_path):
    r"""Is any individual talus harder than the rest?"""
    tags = [t for t in ('3dmatch_euler45', '3dmatch_so3180') if t in runs]
    if not tags:
        return None
    labels = {'3dmatch_euler45': 'rotations up to 45 degrees', '3dmatch_so3180': 'full random SO(3)'}
    colors = [BLUE, ORANGE]

    specimens = sorted({r['file'][:6] for r in runs[tags[0]]['records']})
    means = {}
    for tag in tags:
        by_specimen = {}
        for r in runs[tag]['records']:
            by_specimen.setdefault(r['file'][:6], []).append(r['rmse_mm'])
        means[tag] = by_specimen
    order = sorted(specimens, key=lambda s: np.mean(means[tags[0]][s]))

    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    y = np.arange(len(order))
    for run_idx, tag in enumerate(tags):
        values = [means[tag][s] for s in order]
        offset = (run_idx - (len(tags) - 1) / 2) * 0.3
        for i, vals in enumerate(values):
            ax.plot([min(vals), max(vals)], [y[i] + offset] * 2, color=colors[run_idx], linewidth=1.2, alpha=0.55)
        ax.scatter([np.mean(v) for v in values], y + offset, s=26, color=colors[run_idx],
                   label=labels.get(tag, tag), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel('point RMSE after registration (mm)')
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_title('Per-specimen accuracy, 3DMatch weights', loc='left', fontweight='bold', pad=26)
    ax.text(0, 1.008, 'dot = mean of 3 trials, line = min-max range; specimens sorted by difficulty',
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.legend(fontsize=8, loc='lower right')
    clean_axes(ax, grid_axis='x')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else osp.join(RESULT_DIR, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    runs = load_results()
    if not runs:
        print('no results in ' + RESULT_DIR)
        return
    made = [
        fig_sixdof(runs, osp.join(out_dir, 'sixdof_errors.png')),
        fig_run_comparison(runs, osp.join(out_dir, 'run_comparison.png')),
        fig_inlier_vs_error(runs, osp.join(out_dir, 'inlier_vs_error.png')),
        fig_per_specimen(runs, osp.join(out_dir, 'per_specimen.png')),
    ]
    for path in made:
        if path:
            print('wrote ' + path)


if __name__ == '__main__':
    main()
