r"""Qualitative renders of a self-pair registration: input pair, GeoTransformer's
alignment, and the per-point residual.

Rebuilds the exact pairs `selfpair_eval.py` produced (same seeds), runs the
pre-trained model on the ones requested, and writes one figure per pair.

    python tools/selfpair_render.py --model 3dmatch --cases 0:0 12:1
"""

import argparse
import glob
import importlib
import os
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

from selfpair_eval import (  # noqa: E402
    EXPERIMENTS, DEFAULT_WEIGHTS, DEFAULT_NUM_POINTS, build_pair, normalize_frame,
    apply_transform, registration_error, sixdof_error, make_parser as eval_parser,
)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
BLUE = '#2a78d6'
ORANGE = '#eb6834'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK_2, 'font.size': 9, 'legend.frameon': False,
})


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='3dmatch', choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--weights', default=None)
    parser.add_argument('--data_dir', default=r'C:\Users\esun3\Documents\talus_small')
    parser.add_argument('--pattern', default='*.stl')
    parser.add_argument('--cases', nargs='+', default=['0:0'], help='mesh_index:trial pairs')
    parser.add_argument('--rotation_mode', default='so3', choices=['euler', 'so3'])
    parser.add_argument('--rotation_magnitude', type=float, default=180.0)
    parser.add_argument('--num_points', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7351)
    parser.add_argument('--plot_points', type=int, default=4000, help='points drawn per cloud')
    parser.add_argument('--neighbor_limits', type=int, nargs='+', default=None,
                        help='reuse the limits of a batch run (see its JSON) so numbers match it exactly')
    parser.add_argument('--output_dir', default=osp.join(REPO_DIR, 'output', 'selfpair', 'figures'))
    return parser


def scatter3d(ax, clouds, colors, labels, title, sizes=None):
    for cloud, color, label in zip(clouds, colors, labels):
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1.2, c=color, alpha=0.45,
                   linewidths=0, label=label, depthshade=False)
    ax.set_title(title, fontsize=9.5, color=INK, pad=0)
    limits = np.concatenate(clouds, axis=0)
    center = (limits.max(axis=0) + limits.min(axis=0)) / 2
    span = (limits.max(axis=0) - limits.min(axis=0)).max() / 2 * 1.05
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=35)
    ax.set_box_aspect((1, 1, 1), zoom=1.45)


def main():
    args = make_parser().parse_args()
    if args.num_points is None:
        args.num_points = DEFAULT_NUM_POINTS[args.model]
    if args.weights is None:
        args.weights = osp.join(REPO_DIR, DEFAULT_WEIGHTS[args.model])

    exp_dir = osp.join(REPO_DIR, 'experiments', EXPERIMENTS[args.model])
    sys.path.insert(0, exp_dir)
    cfg = importlib.import_module('config').make_cfg()
    create_model = importlib.import_module('model').create_model
    from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda

    # the pair builder needs the same fields the evaluation script passes around
    pair_args = eval_parser().parse_args(['--model', args.model])
    pair_args.num_points = args.num_points
    pair_args.rotation_mode = args.rotation_mode
    pair_args.rotation_magnitude = args.rotation_magnitude
    pair_args.seed = args.seed

    files = sorted(glob.glob(osp.join(args.data_dir, args.pattern)))
    model = create_model(cfg).cuda()
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=False)['model'])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    written = []
    calib_cache = []
    for case in args.cases:
        mesh_id, trial = (int(x) for x in case.split(':'))
        mesh = trimesh.load(files[mesh_id], process=False)
        center, radius = normalize_frame(mesh, seed=args.seed + mesh_id)
        rng = np.random.default_rng([args.seed, mesh_id, trial])
        data_dict = build_pair(mesh, pair_args, rng, center, radius)

        if args.neighbor_limits is not None:
            neighbor_limits = args.neighbor_limits
        else:
            calib_cache.append(data_dict)
            neighbor_limits = calibrate_neighbors_stack_mode(
                calib_cache, registration_collate_fn_stack_mode, cfg.backbone.num_stages,
                cfg.backbone.init_voxel_size, cfg.backbone.init_radius)
        collated = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius, neighbor_limits)
        with torch.no_grad():
            output_dict = release_cuda(model(to_cuda(collated)))

        gt = data_dict['transform'].astype(np.float64)
        est = np.asarray(output_dict['estimated_transform']).astype(np.float64)
        rre, rte = registration_error(gt, est)
        rot_err, trans_err = sixdof_error(gt, est)
        mm = radius / pair_args.scale

        ref = np.asarray(data_dict['ref_points'], dtype=np.float64)
        src = np.asarray(data_dict['src_points'], dtype=np.float64)
        keep = np.random.default_rng(0).choice(len(ref), min(args.plot_points, len(ref)), replace=False)
        ref_plot, src_plot = ref[keep], src[keep]
        src_aligned = apply_transform(src_plot, est)

        # residual = how far each aligned source point sits from where the gt transform puts it
        residual_mm = np.linalg.norm(apply_transform(src_plot, est) - apply_transform(src_plot, gt), axis=1) * mm

        fig = plt.figure(figsize=(13.5, 4.4))
        ax1 = fig.add_subplot(131, projection='3d')
        scatter3d(ax1, [ref_plot, src_plot], [BLUE, ORANGE], ['reference', 'source'],
                  'input pair (source randomly transformed)')
        ax1.legend(loc='lower left', fontsize=8, markerscale=6, bbox_to_anchor=(0.02, 0.0))

        ax2 = fig.add_subplot(132, projection='3d')
        scatter3d(ax2, [ref_plot, src_aligned], [BLUE, ORANGE], ['reference', 'source aligned'],
                  'after GeoTransformer (no RANSAC)')
        ax2.legend(loc='lower left', fontsize=8, markerscale=6, bbox_to_anchor=(0.02, 0.0))

        ax3 = fig.add_subplot(133, projection='3d')
        pts = apply_transform(src_plot, est)
        sc = ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.6, c=residual_mm, cmap='Blues',
                         vmin=0, vmax=max(residual_mm.max(), 1e-6), linewidths=0, depthshade=False)
        ax3.set_title('residual vs ground truth (mm)', fontsize=9.5, color=INK, pad=0)
        ax3.set_axis_off()
        ax3.view_init(elev=18, azim=35)
        ax3.set_box_aspect((1, 1, 1), zoom=1.45)
        bar = fig.colorbar(sc, ax=ax3, shrink=0.62, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7.5, color=INK_2)

        name = osp.basename(files[mesh_id])[:6]
        fig.suptitle('{}  trial {}  --  RRE {:.2f} deg, RTE {:.3f} mm, mean residual {:.3f} mm'.format(
            name, trial, rre, rte * mm, residual_mm.mean()), x=0.02, ha='left', fontsize=11, fontweight='bold')
        fig.text(0.02, 0.90, '6-DoF residual  rx {:+.2f}  ry {:+.2f}  rz {:+.2f} deg   '
                             'tx {:+.3f}  ty {:+.3f}  tz {:+.3f} mm'.format(
                                 rot_err[0], rot_err[1], rot_err[2],
                                 trans_err[0] * mm, trans_err[1] * mm, trans_err[2] * mm),
                 fontsize=8.5, color=INK_2, ha='left')
        fig.subplots_adjust(left=0.0, right=0.97, top=0.86, bottom=0.0, wspace=0.02)
        out_path = osp.join(args.output_dir, 'alignment_{}_t{}.png'.format(name, trial))
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        written.append(out_path)
        print('wrote {}  (RRE {:.2f} deg, residual {:.3f} mm)'.format(out_path, rre, residual_mm.mean()))

    return written


if __name__ == '__main__':
    main()
