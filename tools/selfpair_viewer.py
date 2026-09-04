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
import json
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
    EXPERIMENTS, DEFAULT_WEIGHTS, DEFAULT_NUM_POINTS, build_pair, normalize_frame, load_pair_meshes,
    apply_transform, registration_error, sixdof_error, ransac_icp, make_parser as eval_parser,
)

REF_COLOR = np.array([0.165, 0.471, 0.839])   # blue
SRC_COLOR = np.array([0.922, 0.408, 0.204])   # orange
CORR_COLOR = np.array([0.42, 0.45, 0.49])
GHOST_COLOR = np.array([0.72, 0.75, 0.78])
RESIDUAL_RAMP = np.array([[0.804, 0.886, 0.984], [0.431, 0.655, 0.925],
                          [0.165, 0.471, 0.839], [0.063, 0.259, 0.506]])

HELP = """
  b       side-by-side before/after  <-> single view
  space   play / pause scrub        1 / 2   input pose / aligned pose
  left    scrub back                right   scrub forward
  r       residual colouring        c       correspondence lines
  n / p   next / previous case      h       this help
"""


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='3dmatch', choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--weights', default=None)
    parser.add_argument('--data_dir', default=osp.join(REPO_DIR, 'output', 'random_taluses'),
                        help='meshes for the dense reference cloud')
    parser.add_argument('--pattern', default='*.stl')
    parser.add_argument('--src_dir', default=osp.join(REPO_DIR, 'output', 'random_taluses', 'paint'),
                        help='meshes for the transformed cloud (the painted region); '
                             'pass "" to transform a full copy of the bone instead')
    parser.add_argument('--noise_sides', default='src', choices=['src', 'both'])
    parser.add_argument('--cases', nargs='+', default=None,
                        help='mesh_index:trial pairs (default: every mesh in the directory)')
    parser.add_argument('--trials', type=int, default=1,
                        help='random trials per mesh to step through (default 1)')
    parser.add_argument('--rotation_mode', default='so3', choices=['euler', 'so3'])
    parser.add_argument('--rotation_magnitude', type=float, default=180.0)
    parser.add_argument('--num_points', type=int, default=None)
    parser.add_argument('--draw_points', type=int, default=6000, help='points drawn per cloud')
    parser.add_argument('--neighbor_limits', type=int, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=7351)
    parser.add_argument('--pair_source', default='patch', choices=['patch', 'landmarks'],
                        help='patch: transformed cloud is a region surface patch; '
                             'landmarks: a few points sampled from each landmark region')
    parser.add_argument('--model_file', default=None, help='SSM h5 with landmark regions (--pair_source landmarks)')
    parser.add_argument('--points_per_region', type=int, default=4)
    parser.add_argument('--ghost', default='on', choices=['on', 'off'],
                        help='draw the whole talus as a faint wireframe shell behind the clouds')
    parser.add_argument('--ghost_faces', type=int, default=1200, help='decimation target for that shell')
    parser.add_argument('--reference', default='full', choices=['full', 'regions'],
                        help='landmarks mode: reference is the whole shape, or only the region vertices')
    parser.add_argument('--points_in_patch', type=int, default=0, help='0 = auto, as in training')
    parser.add_argument('--test_seed', type=int, default=909, help='first shape seed for landmark cases')
    parser.add_argument('--num_cases', type=int, default=10)
    parser.add_argument('--mode', default='compare', choices=['compare', 'scrub'],
                        help='compare: before and after side by side; scrub: one view you animate')
    parser.add_argument('--baseline', default='ransac_icp', choices=['ransac_icp', 'off'],
                        help='third station: RANSAC over the same correspondences, then ICP')
    parser.add_argument('--selftest', action='store_true', help='build everything, skip opening the window')
    parser.add_argument('--snapshot', default=None, help='render each case to PNG(s) at this path instead of opening a window')
    return parser


TEXT_COLOR = np.array([0.09, 0.11, 0.13])
TEXT_UNITS = 13.0  # cap height of Open3D's built-in font, in its own units


def text_mesh(lines, height, color, anchor):
    r"""Multi-line label as a flat mesh in the XY plane, top-left corner at `anchor`.

    The legacy visualizer draws no text, so labels are geometry: they sit in the
    scene and turn with it, which reads like an engraved caption.
    """
    merged = None
    for i, line in enumerate(lines):
        mesh = o3d.t.geometry.TriangleMesh.create_text(line, depth=0.0).to_legacy()
        mesh.scale(height / TEXT_UNITS, center=(0, 0, 0))
        mesh.translate((0.0, -i * height * 1.75, 0.0))
        merged = mesh if merged is None else merged + mesh
    merged.paint_uniform_color(color)
    box = merged.get_axis_aligned_bounding_box()
    merged.translate(np.asarray(anchor, dtype=float)
                     - np.array([box.min_bound[0], box.max_bound[1], 0.0]))
    return merged


def sphere_cloud(points, colors, radius):
    r"""One mesh of small spheres, so a handful of landmark points stay visible."""
    unit = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
    unit.compute_vertex_normals()
    base = np.asarray(unit.vertices)
    triangles = np.asarray(unit.triangles)
    vertices, faces, vertex_colors = [], [], []
    for i, point in enumerate(points):
        vertices.append(base + point)
        faces.append(triangles + i * len(base))
        vertex_colors.append(np.tile(colors[i], (len(base), 1)))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.vstack(vertices)),
        o3d.utility.Vector3iVector(np.vstack(faces)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(vertex_colors))
    mesh.compute_vertex_normals()
    return mesh


def ghost_wireframe(vertices, faces, target_faces, color):
    r"""Faint wireframe shell of the whole bone.

    The legacy renderer has no alpha, so a decimated wireframe stands in for a
    translucent surface: it gives the anatomy for context without hiding the
    points in front of or inside it.
    """
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(faces))
    if target_faces and len(faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(int(target_faces))
    lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lines.paint_uniform_color(color)
    return lines


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
        self.landmarks = None
        if args.pair_source == 'landmarks':
            from train_landmarks import LandmarkShapes, DEFAULT_MODEL
            self.landmarks = LandmarkShapes(args.model_file or DEFAULT_MODEL, '95%', args.points_per_region)
            num_src = 6 * args.points_per_region
            # must match training: 64 points per patch exceeds the whole sparse cloud
            self.cfg.model.num_points_in_patch = args.points_in_patch or max(4, min(64, num_src // 2))
        create_model = importlib.import_module('model').create_model
        from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
        from geotransformer.utils.torch import to_cuda, release_cuda
        self.collate = registration_collate_fn_stack_mode
        self.calibrate = calibrate_neighbors_stack_mode
        self.to_cuda, self.release_cuda = to_cuda, release_cuda

        if self.landmarks is not None:
            # shapes come from the SSM by seed, not from files on disk
            self.files = []
            self.case_specs = [tuple(int(x) for x in c.split(':')) for c in args.cases] if args.cases \
                else [(args.test_seed + i, 0) for i in range(args.num_cases)]
        else:
            self.files = sorted(glob.glob(osp.join(args.data_dir, args.pattern)))
            if not self.files:
                raise RuntimeError('no meshes matching {} in {}'.format(args.pattern, args.data_dir))
            self.case_specs = [tuple(int(x) for x in c.split(':')) for c in args.cases] if args.cases \
                else [(i, t) for i in range(len(self.files)) for t in range(args.trials)]

        self.pair_args = eval_parser().parse_args(['--model', args.model])
        self.pair_args.num_points = args.num_points
        self.pair_args.rotation_mode = args.rotation_mode
        self.pair_args.rotation_magnitude = args.rotation_magnitude
        self.pair_args.seed = args.seed
        self.pair_args.noise_sides = args.noise_sides
        self.pair_args.reference = args.reference

        self.model = create_model(self.cfg).cuda()
        self.model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=False)['model'])
        self.model.eval()
        self.neighbor_limits = args.neighbor_limits
        if self.neighbor_limits is None:
            # a run's limits are part of its result: re-calibrating here would
            # silently produce numbers that differ from the scored ones
            history_path = osp.join(osp.dirname(osp.abspath(args.weights)), 'history.json')
            if osp.exists(history_path):
                with open(history_path) as f:
                    self.neighbor_limits = json.load(f)['neighbor_limits']
                print('neighbor limits {} (from {})'.format(self.neighbor_limits, history_path), flush=True)
        self.cache = {}

    def case(self, index):
        r"""Registered case, computed once and kept."""
        if index in self.cache:
            return self.cache[index]
        mesh_id, trial = self.case_specs[index]
        if self.landmarks is not None:
            data_dict, radius = self.landmarks.pair(mesh_id, self.pair_args)
            name = 'seed {}'.format(mesh_id)
            shape_vertices, _ = self.landmarks.shape(mesh_id)
            shape_mesh = trimesh.Trimesh(shape_vertices, self.landmarks.faces, process=False)
            shape_center, shape_radius = self.landmarks.frame(shape_mesh, mesh_id)
            ghost = ((shape_vertices - shape_center) / shape_radius, np.asarray(self.landmarks.faces))
        else:
            mesh, src_mesh = load_pair_meshes(self.files[mesh_id], self.args.src_dir)
            center, radius = normalize_frame(mesh, seed=self.pair_args.seed + mesh_id)
            rng = np.random.default_rng([self.pair_args.seed, mesh_id, trial])
            data_dict = build_pair(mesh, self.pair_args, rng, center, radius, src_mesh)
            name = osp.splitext(osp.basename(self.files[mesh_id]))[0]
            ghost = ((np.asarray(mesh.vertices) - center) / radius, np.asarray(mesh.faces))

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
        # the two clouds can have different sizes (a region patch is smaller), so thin them separately
        rng_draw = np.random.default_rng(mesh_id)
        ref = ref[rng_draw.choice(len(ref), min(self.args.draw_points, len(ref)), replace=False)]
        src = src[rng_draw.choice(len(src), min(self.args.draw_points, len(src)), replace=False)]
        est_mm, gt_mm = est.copy(), gt.copy()
        est_mm[:3, 3] *= mm
        gt_mm[:3, 3] *= mm

        aligned = apply_transform(src, est_mm)
        residual = np.linalg.norm(aligned - apply_transform(src, gt_mm), axis=1)
        rre, rte = registration_error(gt, est)
        rot_err, trans_err = sixdof_error(gt, est)

        baseline = None
        if self.args.baseline == 'ransac_icp':
            # same correspondences, classical estimator, in the normalized frame
            fit, _ = ransac_icp(np.asarray(data_dict['ref_points'], dtype=np.float64),
                                np.asarray(data_dict['src_points'], dtype=np.float64),
                                np.asarray(output['ref_corr_points'], dtype=np.float64),
                                np.asarray(output['src_corr_points'], dtype=np.float64),
                                mesh=ghost)
            fit_mm = fit.copy()
            fit_mm[:3, 3] *= mm
            b_rre, b_rte = registration_error(gt, fit)
            b_rot, b_trans = sixdof_error(gt, fit)
            baseline = {
                'aligned': apply_transform(src, fit_mm),
                'residual': np.linalg.norm(apply_transform(src, fit_mm) - apply_transform(src, gt_mm), axis=1),
                'rre': float(b_rre), 'rte': float(b_rte * mm),
                'rot_err': b_rot, 'trans_err': b_trans * mm,
            }

        case = {
            'name': name,
            'trial': trial, 'ref': ref, 'src': src, 'aligned': aligned, 'residual': residual,
            'rre': float(rre), 'rte': float(rte * mm), 'time': elapsed,
            'rot_err': rot_err, 'trans_err': trans_err * mm,
            'corr_ref': np.asarray(output['ref_corr_points'], dtype=np.float64) * mm,
            'corr_src': apply_transform(np.asarray(output['src_corr_points'], dtype=np.float64) * mm, est_mm),
            'ghost': (ghost[0] * mm, ghost[1]),
            'baseline': baseline,
        }
        self.cache[index] = case
        return case


def describe(case, index, total):
    def errors(fit):
        # a case with no recorded pose carries NaN errors: say nothing rather than 'nan'
        if not np.isfinite(fit['rre']):
            return ''
        return ('\n    RRE {:.2f} deg  RTE {:.3f} mm  mean residual {:.3f} mm\n'
                '    6-DoF  rx {:+.2f}  ry {:+.2f}  rz {:+.2f} deg   tx {:+.3f}  ty {:+.3f}  tz {:+.3f} mm'
                .format(fit['rre'], fit['rte'], fit['residual'].mean(), *fit['rot_err'], *fit['trans_err']))

    text = '\n[{}/{}] {}  trial {}\n  GeoTransformer  ({:.2f}s on GPU)'.format(
        index + 1, total, case['name'], case['trial'], case['time']) + errors(case)
    for line in case.get('notes', []):
        text += '\n    ' + line
    if case.get('baseline'):
        b = case['baseline']
        text += '\n  RANSAC + ICP' + errors(b)
        for line in b.get('notes', []):
            text += '\n    ' + line
    return text


def run_viewer(session, args, sparse):
    r"""Open the window on `session` (or, with `args.selftest`, just build every case).

    `session` needs `case_specs` and `case(index)`; each case is the dict `Session.case`
    builds. A case may carry `notes` (extra label lines under the GeoTransformer
    numbers) and its baseline may carry `notes` too. `args.snapshot`, when set, renders
    every case to PNG(s) at that path and returns without opening a window.
    """
    state = {'index': 0, 'blend': 0.0 if args.mode == 'compare' else 1.0, 'playing': False,
             'direction': 1.0, 'residual': False, 'corr': False, 'last': time.time(),
             'compare': args.mode == 'compare'}

    # station A (left) is the live one; station B (right) holds the fitted result
    # so before and after can be read at once under a single orbit
    ref_pcd = o3d.geometry.PointCloud()
    src_pcd = o3d.geometry.TriangleMesh() if sparse else o3d.geometry.PointCloud()
    ref_b_pcd = o3d.geometry.PointCloud()
    src_b_pcd = o3d.geometry.TriangleMesh() if sparse else o3d.geometry.PointCloud()
    ref_c_pcd = o3d.geometry.PointCloud()
    src_c_pcd = o3d.geometry.TriangleMesh() if sparse else o3d.geometry.PointCloud()
    third = args.baseline == 'ransac_icp'
    corr_lines = o3d.geometry.LineSet()
    ghost_a = o3d.geometry.LineSet()
    ghost_b = o3d.geometry.LineSet()

    labels = []          # label meshes currently added to the window

    def station_offset(case):
        merged = np.vstack([case['ref'], case['src'], case['aligned']])
        return np.array([1.5 * (merged[:, 0].max() - merged[:, 0].min()), 0.0, 0.0])

    def current():
        return session.case(state['index'])

    def marker_radius(case):
        span = np.ptp(case['ref'], axis=0).max()
        return 0.018 * span

    def refresh_source(vis=None):
        case = current()
        points = case['src'] + (case['aligned'] - case['src']) * state['blend']
        if state['residual'] and state['blend'] > 0.999:
            colors = ramp_colors(case['residual'], np.percentile(case['residual'], 98))
        else:
            colors = np.tile(SRC_COLOR, (len(points), 1))
        if sparse:
            blob = sphere_cloud(points, colors, marker_radius(case))
            src_pcd.vertices = blob.vertices
            src_pcd.triangles = blob.triangles
            src_pcd.vertex_colors = blob.vertex_colors
            src_pcd.vertex_normals = blob.vertex_normals
        else:
            src_pcd.points = o3d.utility.Vector3dVector(points)
            src_pcd.colors = o3d.utility.Vector3dVector(colors)
        if vis is not None:
            vis.update_geometry(src_pcd)

    def paint_station(ref_geom, src_geom, ref_points, src_points, colors, case):
        """Fill one station: its copy of the reference cloud and a transformed cloud."""
        ref_geom.points = o3d.utility.Vector3dVector(ref_points)
        ref_geom.colors = o3d.utility.Vector3dVector(np.tile(REF_COLOR, (len(ref_points), 1)))
        if sparse:
            blob = sphere_cloud(src_points, colors, marker_radius(case))
            src_geom.vertices = blob.vertices
            src_geom.triangles = blob.triangles
            src_geom.vertex_colors = blob.vertex_colors
            src_geom.vertex_normals = blob.vertex_normals
        else:
            src_geom.points = o3d.utility.Vector3dVector(src_points)
            src_geom.colors = o3d.utility.Vector3dVector(colors)

    def clear_station(ref_geom, src_geom):
        ref_geom.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        ref_geom.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        if sparse:
            src_geom.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            src_geom.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
        else:
            src_geom.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            src_geom.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

    def refresh_third(vis=None):
        case = current()
        if state['compare'] and third and case.get('baseline'):
            offset = station_offset(case) * 2
            baseline = case['baseline']
            colors = (ramp_colors(baseline['residual'], np.percentile(baseline['residual'], 98))
                      if state['residual'] else np.tile(SRC_COLOR, (len(baseline['aligned']), 1)))
            paint_station(ref_c_pcd, src_c_pcd, case['ref'] + offset,
                          baseline['aligned'] + offset, colors, case)
        else:
            clear_station(ref_c_pcd, src_c_pcd)
        if vis is not None:
            vis.update_geometry(ref_c_pcd)
            vis.update_geometry(src_c_pcd)

    def refresh_compare(vis=None):
        case = current()
        if state['compare']:
            offset = station_offset(case)
            ref_b_pcd.points = o3d.utility.Vector3dVector(case['ref'] + offset)
            ref_b_pcd.colors = o3d.utility.Vector3dVector(np.tile(REF_COLOR, (len(case['ref']), 1)))
            aligned = case['aligned'] + offset
            colors = (ramp_colors(case['residual'], np.percentile(case['residual'], 98))
                      if state['residual'] else np.tile(SRC_COLOR, (len(aligned), 1)))
            if sparse:
                blob = sphere_cloud(aligned, colors, marker_radius(case))
                src_b_pcd.vertices = blob.vertices
                src_b_pcd.triangles = blob.triangles
                src_b_pcd.vertex_colors = blob.vertex_colors
                src_b_pcd.vertex_normals = blob.vertex_normals
            else:
                src_b_pcd.points = o3d.utility.Vector3dVector(aligned)
                src_b_pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            ref_b_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            ref_b_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            if sparse:
                src_b_pcd.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                src_b_pcd.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
            else:
                src_b_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                src_b_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        if vis is not None:
            vis.update_geometry(ref_b_pcd)
            vis.update_geometry(src_b_pcd)

    def refresh_ghost(vis=None):
        if args.ghost != 'on':
            return
        case = current()
        vertices, faces = case['ghost']
        shell = ghost_wireframe(vertices, faces, args.ghost_faces, GHOST_COLOR)
        ghost_a.points = shell.points
        ghost_a.lines = shell.lines
        ghost_a.colors = shell.colors
        if state['compare']:
            offset = station_offset(case)
            stations = [offset, offset * 2] if (third and case.get('baseline')) else [offset]
            base_points = np.asarray(shell.points)
            base_lines = np.asarray(shell.lines)
            points = np.vstack([base_points + step for step in stations])
            lines = np.vstack([base_lines + i * len(base_points) for i in range(len(stations))])
            ghost_b.points = o3d.utility.Vector3dVector(points)
            ghost_b.lines = o3d.utility.Vector2iVector(lines)
            ghost_b.colors = o3d.utility.Vector3dVector(np.tile(GHOST_COLOR, (len(lines), 1)))
        else:
            ghost_b.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            ghost_b.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        if vis is not None:
            vis.update_geometry(ghost_a)
            vis.update_geometry(ghost_b)

    def refresh_labels(vis=None):
        case = current()
        merged = np.vstack([case['ref'], case['src'], case['aligned']])
        width = merged[:, 0].max() - merged[:, 0].min()
        height = 0.045 * width
        top = np.array([case['ref'][:, 0].min(), merged[:, 1].max() + 0.26 * width,
                        case['ref'][:, 2].mean()])
        def error_lines_for(fit):
            if not np.isfinite(fit['rre']):
                return list(fit.get('notes', []))
            return ['RRE {:.2f} deg   RTE {:.3f} mm'.format(fit['rre'], fit['rte']),
                    'rx {:+.2f}  ry {:+.2f}  rz {:+.2f} deg'.format(*fit['rot_err']),
                    'tx {:+.3f}  ty {:+.3f}  tz {:+.3f} mm'.format(*fit['trans_err'])] + list(fit.get('notes', []))

        error_lines = error_lines_for(case)
        name = '{}  trial {}'.format(case['name'], case['trial'])
        position = '[{}/{}]'.format(state['index'] + 1, len(session.case_specs))

        new = []
        if state['compare']:
            new.append(text_mesh([name, position + '  BEFORE FIT'],
                                 height, TEXT_COLOR, top))
            new.append(text_mesh(['AFTER FIT  (GeoTransformer)'] + error_lines, height, TEXT_COLOR,
                                 top + station_offset(case)))
            if third and case.get('baseline'):
                b = case['baseline']
                new.append(text_mesh(['RANSAC + ICP'] + error_lines_for(b),
                                     height, TEXT_COLOR, top + station_offset(case) * 2))
        else:
            new.append(text_mesh([name] + error_lines, height, TEXT_COLOR, top))

        if vis is not None:
            for old in labels:
                vis.remove_geometry(old, reset_bounding_box=False)
            for mesh in new:
                vis.add_geometry(mesh, reset_bounding_box=False)
        labels[:] = new

    def refresh_correspondences(vis=None):
        case = current()
        if state['corr']:  # an empty LineSet warns every frame, so it is added only when shown
            step = max(1, len(case['corr_ref']) // 240)  # a readable sample of the matches
            a, b = case['corr_ref'][::step], case['corr_src'][::step]
            corr_lines.points = o3d.utility.Vector3dVector(np.vstack([a, b]))
            corr_lines.lines = o3d.utility.Vector2iVector(
                np.stack([np.arange(len(a)), np.arange(len(a)) + len(a)], axis=1))
            corr_lines.colors = o3d.utility.Vector3dVector(np.tile(CORR_COLOR, (len(a), 1)))
        if vis is not None and state['corr']:
            vis.update_geometry(corr_lines)

    def load_case(vis=None):
        case = current()
        ref_pcd.points = o3d.utility.Vector3dVector(case['ref'])
        ref_pcd.colors = o3d.utility.Vector3dVector(np.tile(REF_COLOR, (len(case['ref']), 1)))
        refresh_source(vis)
        refresh_compare(vis)
        refresh_third(vis)
        refresh_ghost(vis)
        refresh_labels(vis)
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

    snapshot = getattr(args, 'snapshot', None)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='GeoTransformer self-pair viewer', width=1440, height=900,
                      visible=not snapshot)
    vis.add_geometry(ref_pcd)
    vis.add_geometry(src_pcd)
    if state['compare']:
        vis.add_geometry(ref_b_pcd)
        vis.add_geometry(src_b_pcd)
        if third:
            refresh_third()
            vis.add_geometry(ref_c_pcd)
            vis.add_geometry(src_c_pcd)
    if args.ghost == 'on':
        refresh_ghost()
        vis.add_geometry(ghost_a)
        if state['compare']:
            vis.add_geometry(ghost_b)
    refresh_labels()
    for mesh in labels:
        vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.988, 0.988, 0.984])
    opt.point_size = 7.0 if len(np.asarray(ref_pcd.points)) < 2000 else 2.5
    opt.mesh_show_back_face = True   # labels are flat meshes; never cull them

    if snapshot:
        # offscreen frames of the opening view, one per case, for reports and quick checks
        stem, ext = osp.splitext(snapshot)
        for i in range(len(session.case_specs)):
            state['index'] = i
            load_case(vis)
            vis.reset_view_point(True)
            vis.poll_events()
            vis.update_renderer()
            case = current()
            path = snapshot if len(session.case_specs) == 1 else '{}_{}_t{}{}'.format(
                stem, case['name'], case['trial'], ext or '.png')
            vis.capture_screen_image(path, do_render=True)
            print('wrote ' + path)
        vis.destroy_window()
        return

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
        if state['residual'] and not state['compare']:
            state['blend'] = 1.0
        refresh_source(vis)
        refresh_compare(vis)
        refresh_third(vis)
        print('  residual colouring {} (0 .. {:.3f} mm at the 98th percentile)'.format(
            'on' if state['residual'] else 'off', np.percentile(current()['residual'], 98)))
        return True

    def toggle_corr(vis):
        state['corr'] = not state['corr']
        if state['corr']:
            refresh_correspondences()
            vis.add_geometry(corr_lines, reset_bounding_box=False)
        else:
            vis.remove_geometry(corr_lines, reset_bounding_box=False)
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
    def toggle_compare(vis):
        state['compare'] = not state['compare']
        if state['compare']:
            state['blend'], state['playing'] = 0.0, False
            vis.add_geometry(ref_b_pcd, reset_bounding_box=False)
            vis.add_geometry(src_b_pcd, reset_bounding_box=False)
            if third:
                vis.add_geometry(ref_c_pcd, reset_bounding_box=False)
                vis.add_geometry(src_c_pcd, reset_bounding_box=False)
            if args.ghost == 'on':
                vis.add_geometry(ghost_b, reset_bounding_box=False)
        else:
            vis.remove_geometry(ref_b_pcd, reset_bounding_box=False)
            vis.remove_geometry(src_b_pcd, reset_bounding_box=False)
            if third:
                vis.remove_geometry(ref_c_pcd, reset_bounding_box=False)
                vis.remove_geometry(src_c_pcd, reset_bounding_box=False)
            if args.ghost == 'on':
                vis.remove_geometry(ghost_b, reset_bounding_box=False)
        refresh_source(vis)
        refresh_compare(vis)
        refresh_third(vis)
        refresh_ghost(vis)
        refresh_labels(vis)
        vis.reset_view_point(True)
        print('  {}'.format('side-by-side: left = before fit, right = after fit'
                            if state['compare'] else 'single view: scrub with space / arrows'))
        return True

    vis.register_key_callback(ord('B'), toggle_compare)
    vis.register_key_callback(ord('H'), lambda v: (print(HELP), True)[1])
    vis.register_animation_callback(animate)

    print(HELP)
    vis.run()
    vis.destroy_window()


def main():
    args = make_parser().parse_args()
    run_viewer(Session(args), args, sparse=args.pair_source == 'landmarks')


if __name__ == '__main__':
    main()
