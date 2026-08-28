r"""Live 3D viewer for self-pair registrations (Open3D window, mouse-orbit).

Rebuilds the same pairs `selfpair_eval.py` scores, runs the pre-trained model on
them, and opens an interactive window where you can scrub the source cloud from
its random pose onto the reference and inspect where the residual sits.

    python tools/selfpair_viewer.py                      # painted patches
    python tools/selfpair_viewer.py --data_dir ../talus_small --pattern "*.stl"

Keys (also printed in the console):
    space   play / pause the alignment scrub      1 / 2   jump to input / aligned
    left    scrub back        right   scrub forward
    r       residual colouring on the source      c       correspondence lines
    n / p   next / previous case                  h       reprint this help
"""

import argparse
import glob
import importlib
import os.path as osp
import sys
import time

import numpy as np
import open3d as o3d
import torch
import trimesh

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

from selfpair_eval import (  # noqa: E402
    EXPERIMENTS, DEFAULT_WEIGHTS, DEFAULT_NUM_POINTS, build_pair, normalize_frame,
    apply_transform, registration_error, sixdof_error, make_parser as eval_parser,
)

REF_COLOR = np.array([0.165, 0.471, 0.839])   # blue
SRC_COLOR = np.array([0.922, 0.408, 0.204])   # orange
CORR_COLOR = np.array([0.42, 0.45, 0.49])
RESIDUAL_RAMP = np.array([[0.804, 0.886, 0.984], [0.431, 0.655, 0.925],
                          [0.165, 0.471, 0.839], [0.063, 0.259, 0.506]])

HELP = """
  space   play / pause scrub        1 / 2   input pose / aligned pose
  left    scrub back                right   scrub forward
  r       residual colouring        c       correspondence lines
  n / p   next / previous case      h       this help
"""


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='3dmatch', choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--weights', default=None)
    parser.add_argument('--data_dir', default=osp.join(REPO_DIR, 'output', 'random_taluses', 'paint'))
    parser.add_argument('--pattern', default='*.stl')
    parser.add_argument('--cases', nargs='+', default=None,
                        help='mesh_index:trial pairs (default: trial 0 of the first 6 meshes)')
    parser.add_argument('--rotation_mode', default='so3', choices=['euler', 'so3'])
    parser.add_argument('--rotation_magnitude', type=float, default=180.0)
    parser.add_argument('--num_points', type=int, default=None)
    parser.add_argument('--draw_points', type=int, default=6000, help='points drawn per cloud')
    parser.add_argument('--neighbor_limits', type=int, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=7351)
    parser.add_argument('--selftest', action='store_true', help='build everything, skip opening the window')
    return parser


def ramp_colors(values, vmax):
    t = np.clip(values / max(vmax, 1e-9), 0, 0.999) * (len(RESIDUAL_RAMP) - 1)
    lo = np.floor(t).astype(int)
    frac = (t - lo)[:, None]
    return RESIDUAL_RAMP[lo] * (1 - frac) + RESIDUAL_RAMP[lo + 1] * frac


class Session:
    r"""Model, cases, and the geometry currently on screen."""

    def __init__(self, args):
        self.args = args
        if args.num_points is None:
            args.num_points = DEFAULT_NUM_POINTS[args.model]
        if args.weights is None:
            args.weights = osp.join(REPO_DIR, DEFAULT_WEIGHTS[args.model])

        exp_dir = osp.join(REPO_DIR, 'experiments', EXPERIMENTS[args.model])
        sys.path.insert(0, exp_dir)
        self.cfg = importlib.import_module('config').make_cfg()
        create_model = importlib.import_module('model').create_model
        from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
        from geotransformer.utils.torch import to_cuda, release_cuda
        self.collate = registration_collate_fn_stack_mode
        self.calibrate = calibrate_neighbors_stack_mode
        self.to_cuda, self.release_cuda = to_cuda, release_cuda

        self.files = sorted(glob.glob(osp.join(args.data_dir, args.pattern)))
        if not self.files:
            raise RuntimeError('no meshes matching {} in {}'.format(args.pattern, args.data_dir))
        self.case_specs = [tuple(int(x) for x in c.split(':')) for c in args.cases] if args.cases \
            else [(i, 0) for i in range(min(6, len(self.files)))]

        self.pair_args = eval_parser().parse_args(['--model', args.model])
        self.pair_args.num_points = args.num_points
        self.pair_args.rotation_mode = args.rotation_mode
        self.pair_args.rotation_magnitude = args.rotation_magnitude
        self.pair_args.seed = args.seed

        self.model = create_model(self.cfg).cuda()
        self.model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=False)['model'])
        self.model.eval()
        self.neighbor_limits = args.neighbor_limits
        self.cache = {}

    def case(self, index):
        r"""Registered case, computed once and kept."""
        if index in self.cache:
            return self.cache[index]
        mesh_id, trial = self.case_specs[index]
        mesh = trimesh.load(self.files[mesh_id], process=False)
        center, radius = normalize_frame(mesh, seed=self.pair_args.seed + mesh_id)
        rng = np.random.default_rng([self.pair_args.seed, mesh_id, trial])
        data_dict = build_pair(mesh, self.pair_args, rng, center, radius)

        if self.neighbor_limits is None:
            self.neighbor_limits = self.calibrate(
                [data_dict], self.collate, self.cfg.backbone.num_stages,
                self.cfg.backbone.init_voxel_size, self.cfg.backbone.init_radius)
        collated = self.collate([data_dict], self.cfg.backbone.num_stages, self.cfg.backbone.init_voxel_size,
                                self.cfg.backbone.init_radius, self.neighbor_limits)
        start = time.time()
        with torch.no_grad():
            output = self.release_cuda(self.model(self.to_cuda(collated)))
        elapsed = time.time() - start

        gt = data_dict['transform'].astype(np.float64)
        est = np.asarray(output['estimated_transform']).astype(np.float64)
        mm = radius  # the pair lives in normalized units; this scales it to millimetres

        ref = np.asarray(data_dict['ref_points'], dtype=np.float64) * mm
        src = np.asarray(data_dict['src_points'], dtype=np.float64) * mm
        keep = np.random.default_rng(mesh_id).choice(
            len(ref), min(self.args.draw_points, len(ref), len(src)), replace=False)
        ref, src = ref[keep], src[keep]
        est_mm, gt_mm = est.copy(), gt.copy()
        est_mm[:3, 3] *= mm
        gt_mm[:3, 3] *= mm

        aligned = apply_transform(src, est_mm)
        residual = np.linalg.norm(aligned - apply_transform(src, gt_mm), axis=1)
        rre, rte = registration_error(gt, est)
        rot_err, trans_err = sixdof_error(gt, est)

        case = {
            'name': osp.splitext(osp.basename(self.files[mesh_id]))[0],
            'trial': trial, 'ref': ref, 'src': src, 'aligned': aligned, 'residual': residual,
            'rre': float(rre), 'rte': float(rte * mm), 'time': elapsed,
            'rot_err': rot_err, 'trans_err': trans_err * mm,
            'corr_ref': np.asarray(output['ref_corr_points'], dtype=np.float64) * mm,
            'corr_src': apply_transform(np.asarray(output['src_corr_points'], dtype=np.float64) * mm, est_mm),
        }
        self.cache[index] = case
        return case


def describe(case, index, total):
    return ('\n[{}/{}] {}  trial {}\n'
            '  RRE {:.2f} deg   RTE {:.3f} mm   mean residual {:.3f} mm   ({:.2f}s on GPU)\n'
            '  6-DoF  rx {:+.2f}  ry {:+.2f}  rz {:+.2f} deg   tx {:+.3f}  ty {:+.3f}  tz {:+.3f} mm'
            .format(index + 1, total, case['name'], case['trial'], case['rre'], case['rte'],
                    case['residual'].mean(), case['time'], *case['rot_err'], *case['trans_err']))


def main():
    args = make_parser().parse_args()
    session = Session(args)

    state = {'index': 0, 'blend': 1.0, 'playing': False, 'direction': -1.0,
             'residual': False, 'corr': False, 'last': time.time()}

    ref_pcd = o3d.geometry.PointCloud()
    src_pcd = o3d.geometry.PointCloud()
    corr_lines = o3d.geometry.LineSet()

    def current():
        return session.case(state['index'])

    def refresh_source(vis=None):
        case = current()
        points = case['src'] + (case['aligned'] - case['src']) * state['blend']
        src_pcd.points = o3d.utility.Vector3dVector(points)
        if state['residual'] and state['blend'] > 0.999:
            colors = ramp_colors(case['residual'], np.percentile(case['residual'], 98))
        else:
            colors = np.tile(SRC_COLOR, (len(points), 1))
        src_pcd.colors = o3d.utility.Vector3dVector(colors)
        if vis is not None:
            vis.update_geometry(src_pcd)

    def refresh_correspondences(vis=None):
        case = current()
        if state['corr']:
            step = max(1, len(case['corr_ref']) // 240)  # a readable sample of the matches
            a, b = case['corr_ref'][::step], case['corr_src'][::step]
            corr_lines.points = o3d.utility.Vector3dVector(np.vstack([a, b]))
            corr_lines.lines = o3d.utility.Vector2iVector(
                np.stack([np.arange(len(a)), np.arange(len(a)) + len(a)], axis=1))
            corr_lines.colors = o3d.utility.Vector3dVector(np.tile(CORR_COLOR, (len(a), 1)))
        else:
            corr_lines.points = o3d.utility.Vector3dVector(
                np.vstack([case['corr_ref'][:1], case['corr_src'][:1]]))
            corr_lines.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        if vis is not None:
            vis.update_geometry(corr_lines)

    def load_case(vis=None):
        case = current()
        ref_pcd.points = o3d.utility.Vector3dVector(case['ref'])
        ref_pcd.colors = o3d.utility.Vector3dVector(np.tile(REF_COLOR, (len(case['ref']), 1)))
        refresh_source(vis)
        refresh_correspondences(vis)
        if vis is not None:
            vis.update_geometry(ref_pcd)
        print(describe(case, state['index'], len(session.case_specs)))

    load_case()
    if args.selftest:
        for i in range(len(session.case_specs)):
            state['index'] = i
            load_case()
        print('\nselftest ok: {} cases built, {} points per cloud'.format(
            len(session.case_specs), len(np.asarray(ref_pcd.points))))
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='GeoTransformer self-pair viewer', width=1440, height=900)
    vis.add_geometry(ref_pcd)
    vis.add_geometry(src_pcd)
    vis.add_geometry(corr_lines)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.988, 0.988, 0.984])
    opt.point_size = 2.5

    def set_blend(value, vis):
        state['blend'] = float(np.clip(value, 0.0, 1.0))
        refresh_source(vis)
        return True

    def step_case(delta, vis):
        state['index'] = (state['index'] + delta) % len(session.case_specs)
        load_case(vis)
        return True

    def toggle_play(vis):
        state['playing'] = not state['playing']
        state['last'] = time.time()
        return True

    def toggle_residual(vis):
        state['residual'] = not state['residual']
        if state['residual']:
            state['blend'] = 1.0
        refresh_source(vis)
        print('  residual colouring {} (0 .. {:.3f} mm at the 98th percentile)'.format(
            'on' if state['residual'] else 'off', np.percentile(current()['residual'], 98)))
        return True

    def toggle_corr(vis):
        state['corr'] = not state['corr']
        refresh_correspondences(vis)
        return True

    def animate(vis):
        if not state['playing']:
            return False
        now = time.time()
        dt, state['last'] = now - state['last'], now
        value = state['blend'] + state['direction'] * dt * 0.55
        if value <= 0.0 or value >= 1.0:
            state['direction'] *= -1
            value = float(np.clip(value, 0.0, 1.0))
        return set_blend(value, vis)

    vis.register_key_callback(ord(' '), toggle_play)
    vis.register_key_callback(ord('1'), lambda v: set_blend(0.0, v))
    vis.register_key_callback(ord('2'), lambda v: set_blend(1.0, v))
    vis.register_key_callback(263, lambda v: set_blend(state['blend'] - 0.05, v))  # left arrow
    vis.register_key_callback(262, lambda v: set_blend(state['blend'] + 0.05, v))  # right arrow
    vis.register_key_callback(ord('R'), toggle_residual)
    vis.register_key_callback(ord('C'), toggle_corr)
    vis.register_key_callback(ord('N'), lambda v: step_case(1, v))
    vis.register_key_callback(ord('P'), lambda v: step_case(-1, v))
    vis.register_key_callback(ord('H'), lambda v: (print(HELP), True)[1])
    vis.register_animation_callback(animate)

    print(HELP)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
