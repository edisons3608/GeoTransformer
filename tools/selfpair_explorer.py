r"""Live, clickable plots of the self-pair results (matplotlib window).

    python tools/selfpair_explorer.py                              # painted vs whole bone
    python tools/selfpair_explorer.py --results output/selfpair    # the real specimens

Left panel  : per-axis 6-DoF residuals, radio buttons switch rotation / translation.
Right panel : inlier ratio against rotation error.
Check boxes toggle runs; clicking any point prints that pair and annotates both panels,
so an outlier in the scatter can be traced straight back to the axis it went wrong in.
"""

import argparse
import glob
import json
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e3e2de'
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#e87ba4', '#eda100']


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=osp.join(REPO_DIR, 'output', 'selfpair_ssm'),
                        help='directory of selfpair_eval JSON results')
    parser.add_argument('--runs', nargs='+', default=None, help='only these tags')
    return parser


def load_runs(results_dir, only=None):
    runs = []
    for path in sorted(glob.glob(osp.join(results_dir, '*.json'))):
        with open(path) as f:
            data = json.load(f)
        tag = data['summary']['tag']
        if only and tag not in only:
            continue
        if 'rot_err_deg' not in data['records'][0]:
            continue  # written before the 6-DoF metrics existed
        runs.append((tag, data['records']))
    if not runs:
        raise RuntimeError('no results with 6-DoF metrics in ' + results_dir)
    return runs


def main():
    args = make_parser().parse_args()
    runs = load_runs(args.results, args.runs)
    colors = {tag: PALETTE[i % len(PALETTE)] for i, (tag, _) in enumerate(runs)}
    visible = {tag: True for tag, _ in runs}
    state = {'metric': 'rotation'}

    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
        'axes.edgecolor': GRID, 'font.size': 9, 'legend.frameon': False,
    })
    fig = plt.figure(figsize=(14, 6.4))
    fig.canvas.manager.set_window_title('Self-pair error explorer')
    ax_axes = fig.add_axes((0.20, 0.13, 0.36, 0.74))
    ax_scatter = fig.add_axes((0.63, 0.13, 0.34, 0.74))
    ax_radio = fig.add_axes((0.015, 0.70, 0.14, 0.16))
    ax_check = fig.add_axes((0.015, 0.42, 0.14, 0.22))
    for ax in (ax_radio, ax_check):
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)

    picks = {}          # artist -> (tag, record, axis index)
    annotations = []

    def clear_annotations():
        while annotations:
            annotations.pop().remove()

    def draw():
        clear_annotations()
        picks.clear()
        ax_axes.clear()
        ax_scatter.clear()
        key = 'rot_err_deg' if state['metric'] == 'rotation' else 'trans_err_mm'
        names = ('rx', 'ry', 'rz') if state['metric'] == 'rotation' else ('tx', 'ty', 'tz')
        unit = 'degrees' if state['metric'] == 'rotation' else 'mm'
        active = [(tag, records) for tag, records in runs if visible[tag]]

        for slot, (tag, records) in enumerate(active):
            errors = np.array([r[key] for r in records])
            color = colors[tag]
            offset = (slot - (len(active) - 1) / 2) * 0.3
            for axis in range(3):
                column = errors[:, axis]
                box = ax_axes.boxplot([column], positions=[axis + offset], widths=0.24,
                                      patch_artist=True, showfliers=False)
                box['boxes'][0].set(facecolor=color + '22', edgecolor=color, linewidth=1.4)
                for part in ('whiskers', 'caps'):
                    for item in box[part]:
                        item.set(color=color, linewidth=1.1)
                box['medians'][0].set(color=color, linewidth=2.2)
                jitter = (np.random.default_rng(axis).random(len(column)) - 0.5) * 0.16
                dots = ax_axes.scatter(axis + offset + jitter, column, s=16, facecolor='none',
                                       edgecolor=color, linewidth=1.0, alpha=.75, picker=5)
                picks[dots] = (tag, records, axis, key, names[axis], unit)

            scatter = ax_scatter.scatter([r['inlier_ratio'] for r in records],
                                         [max(r['rre_deg'], 1e-2) for r in records],
                                         s=22, facecolor='none', edgecolor=color, linewidth=1.1,
                                         alpha=.75, picker=5, label=tag)
            picks[scatter] = (tag, records, None, key, None, unit)

        ax_axes.axhline(0.0, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax_axes.set_xticks(range(3))
        ax_axes.set_xticklabels(names)
        ax_axes.set_ylabel('{} residual ({})'.format(state['metric'], unit))
        ax_axes.set_title('Per-axis 6-DoF residual', loc='left', fontweight='bold')

        ax_scatter.set_yscale('log')
        ax_scatter.axhline(5.0, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax_scatter.annotate('registration failure threshold, 5 deg', xy=(0.02, 5.0),
                            xycoords=('axes fraction', 'data'), xytext=(0, 4),
                            textcoords='offset points', fontsize=8, color=INK_2)
        ax_scatter.set_xlabel('inlier ratio of predicted correspondences')
        ax_scatter.set_ylabel('rotation error RRE (degrees, log)')
        ax_scatter.set_title('Correspondence quality vs error', loc='left', fontweight='bold')
        if active:
            ax_scatter.legend(fontsize=8, loc='lower left')

        for ax in (ax_axes, ax_scatter):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
        fig.canvas.draw_idle()

    def on_pick(event):
        payload = picks.get(event.artist)
        if payload is None:
            return
        tag, records, axis, key, axis_name, unit = payload
        record = records[event.ind[0]]
        clear_annotations()
        text = ('{}  {} trial {}\n'
                'RRE {:.2f} deg   RMSE {:.3f} mm   inliers {:.0f}%\n'
                'rot {:+.2f} {:+.2f} {:+.2f} deg   trans {:+.3f} {:+.3f} {:+.3f} mm'.format(
                    tag, record['file'], record['trial'], record['rre_deg'], record['rmse_mm'],
                    100 * record['inlier_ratio'], *record['rot_err_deg'], *record['trans_err_mm']))
        print('\n' + text)
        note = ax_scatter.annotate(
            '{} t{}'.format(record['file'], record['trial']),
            xy=(record['inlier_ratio'], max(record['rre_deg'], 1e-2)),
            xytext=(12, 14), textcoords='offset points', fontsize=8.5, color=INK,
            bbox=dict(boxstyle='round,pad=0.35', facecolor=SURFACE, edgecolor=colors[tag]),
            arrowprops=dict(arrowstyle='-', color=colors[tag]))
        annotations.append(note)
        # mark the same pair in every axis of the left panel
        for a in range(3):
            marker = ax_axes.plot([a], [record[key][a]], marker='o', markersize=9, markerfacecolor='none',
                                  markeredgecolor=colors[tag], markeredgewidth=1.8)[0]
            annotations.append(marker)
        fig.canvas.draw_idle()

    radio = RadioButtons(ax_radio, ('rotation', 'translation'), active=0)
    check = CheckButtons(ax_check, [tag for tag, _ in runs], [True] * len(runs))
    for label, (tag, _) in zip(check.labels, runs):
        label.set_color(colors[tag])
        label.set_fontsize(8.5)

    def on_metric(label):
        state['metric'] = label
        draw()

    def on_toggle(label):
        visible[label] = not visible[label]
        draw()

    radio.on_clicked(on_metric)
    check.on_clicked(on_toggle)
    fig.canvas.mpl_connect('pick_event', on_pick)

    fig.text(0.015, 0.93, 'Self-pair error explorer', fontsize=13, fontweight='bold')
    fig.text(0.015, 0.90, osp.relpath(args.results, REPO_DIR), fontsize=8.5, color=INK_2)
    fig.text(0.015, 0.34, 'click any point to trace\nthat pair across both panels',
             fontsize=8.5, color=INK_2)
    draw()
    plt.show()


if __name__ == '__main__':
    main()
