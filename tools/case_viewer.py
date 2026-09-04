r"""Live 3D viewer for digitized landmark cases on a patient talus (Open3D window, mouse-orbit).

A case folder holds the patient's `talus.stl` and one CSV per scenario: the ~27
points a surgeon digitized inside the six landmark regions, in the tracker frame
(hundreds of millimetres from the CT frame the STL lives in). The six-region-to-
full-cloud model (`train_landmarks.py`, run `landmarks24_rotref`) registers each
scenario to the bone, and the window is the one `selfpair_viewer.py` opens, with
the same keys: before the fit, GeoTransformer's fit, and RANSAC + ICP side by side.

    python tools/case_viewer.py                                        # S260655_LEFT
    python tools/case_viewer.py --case_dir <folder> --trials 3          # random start poses too
    python tools/case_viewer.py --snapshot output/cases/S260655_LEFT/view.png

There is no recorded pose, so every fit is judged by the distance of its points to the bone surface.
"""

import argparse
import glob
import importlib
import json
import os
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
    EXPERIMENTS, normalize_frame, sample_cloud, random_transform, inverse_transform, apply_transform, ransac_icp,
)
from selfpair_viewer import run_viewer  # noqa: E402

DEFAULT_CASE_DIR = r'C:\Users\esun3\OneDrive - Stryker\Documents\ForEdison\S260655_LEFT'
DEFAULT_WEIGHTS = osp.join(REPO_DIR, 'output', 'landmark_bench', 'landmarks24_rotref', 'best.pth.tar')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_dir', default=DEFAULT_CASE_DIR, help='folder with talus.stl and the scenario CSVs')
    parser.add_argument('--pattern', default='*.csv')
    parser.add_argument('--mesh', default=None, help='the bone (default: talus.stl in the case folder)')
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS, help='six-region-to-full-cloud checkpoint')
    parser.add_argument('--model', default='3dmatch', choices=list(EXPERIMENTS.keys()),
                        help='experiment the checkpoint was fine-tuned from')
    parser.add_argument('--num_points', type=int, default=20000, help='reference samples, as in training')
    parser.add_argument('--points_in_patch', type=int, default=0, help='0 = the value the checkpoint trained with')
    parser.add_argument('--neighbor_limits', type=int, nargs='+', default=None,
                        help='default: the limits recorded next to the checkpoint')
    parser.add_argument('--cases', nargs='+', default=None, help='scenario_index:trial pairs (default: all)')
    parser.add_argument('--trials', type=int, default=1,
                        help='trial 0 is the digitized cloud as is; later trials start it from a random pose')
    parser.add_argument('--rotation_mode', default='so3', choices=['euler', 'so3'])
    parser.add_argument('--rotation_magnitude', type=float, default=180.0)
    parser.add_argument('--translation_magnitude', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=7351)
    parser.add_argument('--baseline', default='ransac_icp', choices=['ransac_icp', 'off'])
    parser.add_argument('--hypotheses', type=int, default=40,
                        help='RANSAC runs per case; the one sitting closest to the bone surface wins')
    parser.add_argument('--ghost', default='on', choices=['on', 'off'])
    parser.add_argument('--ghost_faces', type=int, default=1200)
    parser.add_argument('--mode', default='compare', choices=['compare', 'scrub'])
    parser.add_argument('--out_dir', default=None,
                        help='results JSON goes here (default: output/cases/<case folder name>)')
    parser.add_argument('--selftest', action='store_true', help='build every case, write results, no window')
    parser.add_argument('--snapshot', default=None, help='render each case to PNG(s) at this path, no window')
    return parser


class CaseSession:
    r"""Model, the bone, the digitized scenarios, and the registered cases."""

    def __init__(self, args):
        self.args = args
        exp_dir = osp.join(REPO_DIR, 'experiments', EXPERIMENTS[args.model])
        sys.path.insert(0, exp_dir)
        self.cfg = importlib.import_module('config').make_cfg()
        create_model = importlib.import_module('model').create_model
        from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
        from geotransformer.utils.torch import to_cuda, release_cuda
        self.collate, self.calibrate = registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
        self.to_cuda, self.release_cuda = to_cuda, release_cuda

        # the checkpoint's own training record fixes the grouping and neighbour limits
        trained = {}
        history_path = osp.join(osp.dirname(osp.abspath(args.weights)), 'history.json')
        if osp.exists(history_path):
            with open(history_path) as f:
                trained = json.load(f)
        self.neighbor_limits = args.neighbor_limits or trained.get('neighbor_limits')
        # the grouping has to be the one the checkpoint trained with, so prefer the value
        # it recorded; only fall back to the rule when the run did not set one
        trained_patch = trained.get('args', {}).get('points_in_patch') or 0
        trained_src = 6 * trained.get('args', {}).get('points_per_region', 4)
        self.cfg.model.num_points_in_patch = (args.points_in_patch or trained_patch
                                              or max(4, min(64, trained_src // 2)))
        print('weights {}\npoints per superpoint patch {}, neighbor limits {}'.format(
            args.weights, self.cfg.model.num_points_in_patch, self.neighbor_limits), flush=True)

        self.model = create_model(self.cfg).cuda()
        self.model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=False)['model'])
        self.model.eval()

        self.out_dir = args.out_dir or osp.join(REPO_DIR, 'output', 'cases',
                                                osp.basename(osp.normpath(args.case_dir)))
        self.mesh_path = args.mesh or osp.join(args.case_dir, 'talus.stl')
        mesh = trimesh.load(self.mesh_path, process=False)
        mesh.merge_vertices()                      # STL triangles share no vertices until merged
        self.mesh = mesh
        self.center, self.radius = normalize_frame(mesh, seed=args.seed)
        self.ghost = ((np.asarray(mesh.vertices) - self.center) / self.radius, np.asarray(mesh.faces))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)))))

        self.files = sorted(glob.glob(osp.join(args.case_dir, args.pattern)))
        if not self.files:
            raise RuntimeError('no scenarios matching {} in {}'.format(args.pattern, args.case_dir))
        self.clouds = [np.loadtxt(path, delimiter=',', ndmin=2)[:, :3].astype(np.float64) for path in self.files]
        self.names = [scenario_label(path) for path in self.files]
        self.case_specs = [tuple(int(x) for x in c.split(':')) for c in args.cases] if args.cases \
            else [(i, t) for i in range(len(self.files)) for t in range(args.trials)]
        print('{}: {} scenarios, {} points each, bone radius {:.1f} mm'.format(
            osp.basename(osp.normpath(args.case_dir)), len(self.files),
            '/'.join(sorted({str(len(c)) for c in self.clouds})), self.radius), flush=True)

        # one dense reference cloud shared by every scenario, like one CT for the whole case
        rng = np.random.default_rng([args.seed, 0])
        self.ref_points = sample_cloud(mesh, args.num_points, self.center, self.radius, rng)
        # training puts each cloud exactly on its own centroid; the frame's centre comes
        # from a different surface sample, so fold the residual into it and the ghost,
        # leaving the reference cloud exactly at the origin as the source already is
        shift = self.ref_points.mean(axis=0)
        self.ref_points = self.ref_points - shift
        self.center = self.center + self.radius * shift
        self.ghost = ((np.asarray(mesh.vertices) - self.center) / self.radius, np.asarray(mesh.faces))

        self.cache = {}

    def surface_distance_mm(self, points_mm):
        return self.scene.compute_distance(o3d.core.Tensor(np.asarray(points_mm, dtype=np.float32))).numpy()

    def to_mm(self, normalized):
        return normalized * self.radius + self.center

    def case(self, index):
        if index in self.cache:
            return self.cache[index]
        scenario, trial = self.case_specs[index]
        cloud = self.clouds[scenario]
        src_center = cloud.mean(axis=0)
        # the digitized cloud sits in its own frame: centre it, scale it like the bone
        centered = (cloud - src_center) / self.radius
        pose = np.eye(4)
        if trial > 0:
            rng = np.random.default_rng([self.args.seed, scenario, trial])
            pose = random_transform(rng, self.args.rotation_mode, self.args.rotation_magnitude,
                                    self.args.translation_magnitude)
        src_points = apply_transform(centered, inverse_transform(pose))

        gt = np.eye(4)                    # no recorded pose to compare against

        data_dict = {
            'ref_points': self.ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((len(self.ref_points), 1), dtype=np.float32),
            'src_feats': np.ones((len(src_points), 1), dtype=np.float32),
            'transform': gt.astype(np.float32),
        }
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
        est = np.asarray(output['estimated_transform']).astype(np.float64)

        mm = self.radius
        src = src_points * mm
        est_mm = est.copy()
        est_mm[:3, 3] *= mm

        def score(fit):
            r"""Everything the viewer shows for one fit, plus the surface distance and full pose."""
            fit_mm = fit.copy()
            fit_mm[:3, 3] *= mm
            aligned = apply_transform(src, fit_mm)
            surface = self.surface_distance_mm(self.to_mm(apply_transform(src_points, fit)))
            # nothing to compare against: the residual shown is distance to the surface
            residual, rre, rte = surface, float('nan'), float('nan')
            rot, trans = np.full(3, np.nan), np.full(3, np.nan)
            # tracker -> CT, in millimetres: undo the centring, apply the fit, leave the bone's frame
            whole = fit @ inverse_transform(pose)
            tracker_to_ct = np.eye(4)
            tracker_to_ct[:3, :3] = whole[:3, :3]
            tracker_to_ct[:3, 3] = self.center + mm * whole[:3, 3] - whole[:3, :3] @ src_center
            return {
                'aligned': aligned, 'residual': residual,
                'rre': float(rre), 'rte': float(rte * mm), 'rot_err': rot, 'trans_err': trans * mm,
                'surface_mm': surface, 'tracker_to_ct': tracker_to_ct,
                'notes': ['surface RMS {:.2f} mm   max {:.2f} mm'.format(
                    np.sqrt(np.mean(surface ** 2)), surface.max())],
            }

        result = score(est)
        baseline = None
        if self.args.baseline == 'ransac_icp':
            # Open3D's correspondence RANSAC takes no seed, so repeat runs land in
            # different basins. Distance to the bone surface separates a correct fit
            # from a wrong one by roughly five times and needs no ground truth, so it
            # picks the winner: the pose that actually sits on the anatomy.
            best_fit, best_rms = None, float('inf')
            for _ in range(max(1, self.args.hypotheses)):
                fit, _ = ransac_icp(np.asarray(data_dict['ref_points'], dtype=np.float64), src_points,
                                    np.asarray(output['ref_corr_points'], dtype=np.float64),
                                    np.asarray(output['src_corr_points'], dtype=np.float64), mesh=self.ghost)
                surface = self.surface_distance_mm(self.to_mm(apply_transform(src_points, fit)))
                rms = float(np.sqrt(np.mean(surface ** 2)))
                if rms < best_rms:
                    best_fit, best_rms = fit, rms
            baseline = score(best_fit)
            if self.args.hypotheses > 1:
                baseline['notes'].append('best of {} RANSAC hypotheses'.format(self.args.hypotheses))

        case = dict(result, **{
            'name': self.names[scenario], 'trial': trial, 'time': elapsed,
            'ref': self.ref_points * mm, 'src': src,
            'corr_ref': np.asarray(output['ref_corr_points'], dtype=np.float64) * mm,
            'corr_src': apply_transform(np.asarray(output['src_corr_points'], dtype=np.float64) * mm, est_mm),
            'ghost': (self.ghost[0] * mm, self.ghost[1]),
            'baseline': baseline,
        })
        self.cache[index] = case
        return case

    def summarise(self, out_dir):
        r"""Build every case, print one line per case, write the numbers and poses to JSON."""
        rows = [self.case(i) for i in range(len(self.case_specs))]
        third = any(r.get('baseline') for r in rows)
        head = '{:<10}{:>3}   {:>10}{:>10}'.format('scenario', 't', 'surf RMS', 'max mm')
        if third:
            head += '   |{:>10}{:>10}'.format('surf RMS', 'max mm')
        print('\n{:>36}{}'.format('GeoTransformer', '        RANSAC + ICP' if third else ''))
        print(head)
        records = []
        for (scenario, trial), case in zip(self.case_specs, rows):
            line = '{:<10}{:>3}   {:>10.2f}{:>10.2f}'.format(
                case['name'], trial, np.sqrt(np.mean(case['surface_mm'] ** 2)), case['surface_mm'].max())
            record = {'scenario': case['name'], 'file': osp.basename(self.files[scenario]), 'trial': trial,
                      'num_points': len(case['src']), 'geotransformer': pose_record(case)}
            if case.get('baseline'):
                b = case['baseline']
                line += '   |{:>10.2f}{:>10.2f}'.format(
                    np.sqrt(np.mean(b['surface_mm'] ** 2)), b['surface_mm'].max())
                record['ransac_icp'] = pose_record(b)
            print(line)
            records.append(record)

        def recall(key):
            picked = [r[key] for r in records if key in r]
            return {'n': len(picked),
                    'surface_rms_p50': float(np.median([p['surface_rms_mm'] for p in picked])),
                    'under_0p5mm': int(sum(p['surface_rms_mm'] < 0.5 for p in picked))}

        summary = {'geotransformer': recall('geotransformer')}
        if third:
            summary['ransac_icp'] = recall('ransac_icp')
        for key, v in summary.items():
            print('{:<16} median surface RMS {:.2f} mm   under 0.5 mm: {} of {}'.format(
                key, v['surface_rms_p50'], v['under_0p5mm'], v['n']))
        os.makedirs(out_dir, exist_ok=True)
        path = osp.join(out_dir, 'case_results.json')
        with open(path, 'w') as f:
            json.dump({'case_dir': self.args.case_dir, 'weights': self.args.weights, 'args': vars(self.args),
                       'radius_mm': float(self.radius), 'center_mm': self.center.tolist(),
                       'summary': summary, 'records': records}, f, indent=2)
        print('results in ' + path, flush=True)
        self.write_transforms(out_dir, records)

    def write_transforms(self, out_dir, records):
        r"""The poses on their own, in the frames the case actually lives in.

        Everything the network sees is centred and scaled, so its transform is not
        usable outside this script. What is written here maps the raw CSV coordinates
        to the STL's own coordinates in millimetres, with the centring on both sides
        already undone, and is what you would hand to anything downstream.
        """
        # the network's own pose is not usable on this data; the stored pose is the
        # refined one, chosen from the RANSAC hypotheses by distance to the surface
        method = 'ransac_icp' if any('ransac_icp' in r for r in records) else 'geotransformer'
        scenarios = []
        for record in records:
            scenarios.append({'scenario': record['scenario'], 'file': record['file'],
                              'trial': record['trial'], 'num_points': record['num_points'],
                              'tracker_to_ct': record[method]['tracker_to_ct']})

        document = {
            'case_dir': self.args.case_dir, 'mesh': self.mesh_path, 'weights': self.args.weights,
            'hypotheses': self.args.hypotheses, 'method': method, 'units': 'millimetres',
            'convention': 'row-major 4x4; p_ct = M[:3,:3] @ p_tracker + M[:3,3], '
                          'p_tracker being a raw row of the scenario CSV',
            'note': 'the centring the network needs (each cloud on its own centroid) is '
                    'already undone in these matrices',
            'scenarios': scenarios,
        }
        path = osp.join(out_dir, 'transforms.json')
        with open(path, 'w') as f:
            json.dump(document, f, indent=2)
        print('transforms in {}  ({} scenarios)'.format(path, len(scenarios)), flush=True)

def pose_record(fit):
    return {'surface_rms_mm': float(np.sqrt(np.mean(fit['surface_mm'] ** 2))),
            'surface_max_mm': float(fit['surface_mm'].max()),
            'surface_mm': [float(x) for x in fit['surface_mm']],
            'tracker_to_ct': fit['tracker_to_ct'].tolist()}


def scenario_label(path):
    r"""`S260655_LEFT_LC_1_TALUS_Surface.csv` -> `LC_1`; other names keep their stem."""
    stem = osp.splitext(osp.basename(path))[0]
    parts = stem.split('_')
    if len(parts) >= 5 and parts[-1].lower() == 'surface':
        return '_'.join(parts[2:-2])
    return stem


def main():
    args = make_parser().parse_args()
    session = CaseSession(args)
    session.summarise(session.out_dir)
    if args.selftest:
        return
    run_viewer(session, args, sparse=True)


if __name__ == '__main__':
    main()
