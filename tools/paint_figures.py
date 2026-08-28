r"""Figures for the painted-region experiment.

    python tools/paint_figures.py

Writes two PNGs to output/selfpair_ssm/figures:
  paint_region.png  - where the painted region sits on the generated shapes
  paint_vs_full.png - per-axis 6-DoF error, painted patch against the whole bone
"""

import json
import os
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

from generate_random_taluses import load_model, painted_faces, resolve_num_modes, sample_coefficients  # noqa: E402

RESULT_DIR = osp.join(REPO_DIR, 'output', 'selfpair_ssm')
OUT_DIR = osp.join(RESULT_DIR, 'figures')

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e3e2de'
BONE = '#c8d3de'
PAINT = '#eb6834'
BLUE = '#2a78d6'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.edgecolor': GRID, 'font.size': 9, 'legend.frameon': False,
})


def shaded_faces(verts, faces, base_colors, azim, elev):
    """Lambert-shaded per-face colours; one collection so depth sorting is correct."""
    tris = verts[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    a, e = np.radians(azim), np.radians(elev)
    light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    lambert = 0.42 + 0.58 * np.clip(normals @ light, 0, 1)
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in base_colors])
    return tris, np.clip(rgb * lambert[:, None], 0, 1)


def fig_paint_region(model, vertices, paint_faces, out_path):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = model['faces']
    painted = set(map(tuple, np.sort(paint_faces, axis=1).tolist()))
    is_paint = np.array([tuple(f) in painted for f in np.sort(faces, axis=1).tolist()])
    colors = np.where(is_paint, PAINT, BONE)

    # look at the region head-on: aim the camera down the mean paint normal
    mean_verts = vertices[0]
    to_paint = mean_verts[np.unique(paint_faces)].mean(axis=0) - mean_verts.mean(axis=0)
    azim = np.degrees(np.arctan2(to_paint[1], to_paint[0]))
    elev = np.degrees(np.arcsin(to_paint[2] / np.linalg.norm(to_paint)))

    fig = plt.figure(figsize=(3.1 * len(vertices), 3.5))
    for i, verts in enumerate(vertices):
        ax = fig.add_subplot(1, len(vertices), i + 1, projection='3d')
        tris, facecolors = shaded_faces(verts, faces, colors, azim, elev)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=facecolors, edgecolors='none',
                                             linewidths=0, zsort='average'))
        center = verts.mean(axis=0)
        span = np.abs(verts - center).max()
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1), zoom=1.5)
        ax.set_title('sample {:03d}'.format(i - 1) if i else 'mean shape', fontsize=9, color=INK_2, pad=0)
    fig.suptitle('The painted region, carried onto each sampled shape by vertex correspondence',
                 x=0.015, ha='left', fontsize=11, fontweight='bold')
    fig.text(0.015, 0.90, '74 paint spheres of r = 4.33 mm found on the mean shape -> 1153 of 6002 vertices, '
                          '2148 faces, ~20% of the surface; only this patch is fed to the registration',
             fontsize=8.5, color=INK_2, ha='left')
    fig.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.0, wspace=0.0)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


def style_box(bp, color):
    for box in bp['boxes']:
        box.set(facecolor=color + '33', edgecolor=color, linewidth=1.4)
    for part in ('whiskers', 'caps'):
        for item in bp[part]:
            item.set(color=color, linewidth=1.2)
    for median in bp['medians']:
        median.set(color=color, linewidth=2.0)
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=3, markerfacecolor='none', markeredgecolor=color, alpha=0.7)


def fig_paint_vs_full(out_path):
    runs = {}
    for tag in ('ssm_so3180', 'paint_so3180'):
        with open(osp.join(RESULT_DIR, tag + '.json')) as f:
            runs[tag] = json.load(f)['records']
    labels = {'ssm_so3180': 'whole bone', 'paint_so3180': 'painted region only'}
    colors = {'ssm_so3180': BLUE, 'paint_so3180': PAINT}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, names, ylabel in ((axes[0], 'rot_err_deg', ('rx', 'ry', 'rz'), 'rotation residual (degrees)'),
                                   (axes[1], 'trans_err_mm', ('tx', 'ty', 'tz'), 'translation residual (mm)')):
        for run_idx, tag in enumerate(runs):
            errors = np.array([r[key] for r in runs[tag]])
            positions = np.arange(3) + (run_idx - 0.5) * 0.32
            bp = ax.boxplot([errors[:, i] for i in range(3)], positions=positions, widths=0.26,
                            patch_artist=True, showfliers=True)
            style_box(bp, colors[tag])
            bp['boxes'][0].set_label(labels[tag])
        ax.axhline(0.0, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(names)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], color=colors[t], linewidth=3) for t in runs]
    fig.legend(handles, [labels[t] for t in runs], loc='upper right',
               bbox_to_anchor=(0.99, 0.945), fontsize=8.5, ncol=2)
    fig.suptitle('Registering 20% of the bone widens the error, mostly about rx and ry',
                 x=0.02, ha='left', fontsize=11, fontweight='bold')
    fig.text(0.02, 0.90, '10 SSM-generated shapes x 3 trials, full random SO(3), 3DMatch weights',
             fontsize=8.5, color=INK_2, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    from generate_random_taluses import DEFAULT_MODEL
    model = load_model(DEFAULT_MODEL)
    paint_faces, _ = painted_faces(model)
    scale = 1.0 / model['attrs']['norm_scale']

    num_modes = resolve_num_modes('95%', model['eigenvalues'])
    rng = np.random.default_rng(0)
    coefficients = sample_coefficients(rng, 10, num_modes, 3.0, 'gaussian')
    deltas = (coefficients * model['sigmas'][:num_modes]) @ model['pcs'][:num_modes]
    samples = (model['mean'] + deltas).reshape(10, -1, 3) * scale
    shapes = [model['mean'].reshape(-1, 3) * scale, samples[0], samples[1], samples[2]]

    os.makedirs(OUT_DIR, exist_ok=True)
    print('wrote ' + fig_paint_region(model, shapes, paint_faces, osp.join(OUT_DIR, 'paint_region.png')))
    print('wrote ' + fig_paint_vs_full(osp.join(OUT_DIR, 'paint_vs_full.png')))


if __name__ == '__main__':
    main()
