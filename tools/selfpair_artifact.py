r"""Build the interactive self-pair report: exports the benchmark records plus a
few registered point-cloud pairs, and injects them into the HTML template.

    python tools/selfpair_artifact.py

Writes output/selfpair/talus_report.html (self-contained, no external assets).
"""

import argparse
import glob
import importlib
import json
import os
import os.path as osp
import sys

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

RESULT_DIR = osp.join(REPO_DIR, 'output', 'selfpair')
TEMPLATE = osp.join(REPO_DIR, 'tools', 'selfpair_report_template.html')

# (mesh index, trial, rotation mode, label) -- picked from the batch results
CASES = [
    (3, 0, 'so3', 'Typical pair'),
    (12, 1, 'so3', 'Worst pair of 93'),
    (30, 1, 'so3', 'Best pair of 93'),
    (0, 0, 'euler', 'Rotation under 45 degrees'),
]
NEIGHBOR_LIMITS = {'so3': [47, 36, 38, 41], 'euler': [47, 36, 38, 41]}


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=r'C:\Users\esun3\Documents\talus_small')
    parser.add_argument('--pattern', default='*.stl')
    parser.add_argument('--viewer_points', type=int, default=2400, help='points per cloud in the 3D viewer')
    parser.add_argument('--seed', type=int, default=7351)
    parser.add_argument('--out', default=osp.join(RESULT_DIR, 'talus_report.html'))
    return parser


def load_runs():
    runs = []
    for path in sorted(glob.glob(osp.join(RESULT_DIR, '*.json'))):
        with open(path) as f:
            data = json.load(f)
        summary = data['summary']
        runs.append({
            'tag': summary['tag'],
            'model': summary['model'],
            'rotation': 'so3' if 'so3' in summary['tag'] else 'euler',
            'num_points': summary['args']['num_points'],
            'summary': {k: summary[k] for k in
                        ('num_pairs', 'rre_deg', 'rte_mm', 'rmse_mm', 'inlier_ratio_mean',
                         'recall_1deg_1mm', 'recall_5deg_2mm', 'mean_time_s')
                        if k in summary},
            'sixdof': summary.get('sixdof'),
            'records': [{
                'file': r['file'][:6],
                'trial': r['trial'],
                'rre': round(r['rre_deg'], 4),
                'rte': round(r['rte_mm'], 4),
                'rmse': round(r['rmse_mm'], 4),
                'ir': round(r['inlier_ratio'], 4),
                'corr': r['num_corr'],
                'rot': [round(x, 4) for x in r['rot_err_deg']] if 'rot_err_deg' in r else None,
                'trans': [round(x, 4) for x in r['trans_err_mm']] if 'trans_err_mm' in r else None,
            } for r in data['records']],
        })
    return runs


def export_cases(args):
    exp_dir = osp.join(REPO_DIR, 'experiments', EXPERIMENTS['3dmatch'])
    sys.path.insert(0, exp_dir)
    cfg = importlib.import_module('config').make_cfg()
    create_model = importlib.import_module('model').create_model
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda

    files = sorted(glob.glob(osp.join(args.data_dir, args.pattern)))
    model = create_model(cfg).cuda()
    model.load_state_dict(torch.load(osp.join(REPO_DIR, DEFAULT_WEIGHTS['3dmatch']),
                                     map_location='cpu', weights_only=False)['model'])
    model.eval()

    cases = []
    for mesh_id, trial, mode, label in CASES:
        pair_args = eval_parser().parse_args(['--model', '3dmatch'])
        pair_args.num_points = DEFAULT_NUM_POINTS['3dmatch']
        pair_args.rotation_mode = mode
        pair_args.rotation_magnitude = 180.0 if mode == 'so3' else 45.0
        pair_args.seed = args.seed

        mesh = trimesh.load(files[mesh_id], process=False)
        center, radius = normalize_frame(mesh, seed=args.seed + mesh_id)
        rng = np.random.default_rng([args.seed, mesh_id, trial])
        data_dict = build_pair(mesh, pair_args, rng, center, radius)

        collated = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius, NEIGHBOR_LIMITS[mode])
        with torch.no_grad():
            output_dict = release_cuda(model(to_cuda(collated)))

        gt = data_dict['transform'].astype(np.float64)
        est = np.asarray(output_dict['estimated_transform']).astype(np.float64)
        rre, rte = registration_error(gt, est)
        rot_err, trans_err = sixdof_error(gt, est)

        # everything the page draws is in millimetres, so convert here once
        mm = radius
        ref = np.asarray(data_dict['ref_points'], dtype=np.float64) * mm
        src = np.asarray(data_dict['src_points'], dtype=np.float64) * mm
        pick = np.random.default_rng(mesh_id).choice(
            len(ref), min(args.viewer_points, len(ref)), replace=False)
        est_mm, gt_mm = est.copy(), gt.copy()
        est_mm[:3, 3] *= mm
        gt_mm[:3, 3] *= mm

        cases.append({
            'label': label,
            'specimen': osp.basename(files[mesh_id])[:6],
            'trial': trial,
            'rotation': mode,
            'ref': [[round(v, 3) for v in p] for p in ref[pick]],
            'src': [[round(v, 3) for v in p] for p in src[pick[:len(pick)]]],
            'est': [[round(v, 6) for v in row] for row in est_mm],
            'gt': [[round(v, 6) for v in row] for row in gt_mm],
            'rre': round(float(rre), 4),
            'rte': round(float(rte * mm), 4),
            'ir': round(float((np.linalg.norm(
                np.asarray(output_dict['ref_corr_points'], dtype=np.float64)
                - apply_transform(np.asarray(output_dict['src_corr_points'], dtype=np.float64), gt),
                axis=1) < cfg.eval.acceptance_radius).mean()), 4),
            'corr': int(output_dict['corr_scores'].shape[0]),
            'rot_err': [round(float(x), 4) for x in rot_err],
            'trans_err': [round(float(x * mm), 4) for x in trans_err],
        })
        print('exported {}  {}  RRE {:.2f} deg'.format(label, cases[-1]['specimen'], rre))
    return cases


def main():
    args = make_parser().parse_args()
    payload = {'runs': load_runs(), 'cases': export_cases(args)}
    with open(TEMPLATE, encoding='utf-8') as f:
        html = f.read()
    html = html.replace('/*__DATA__*/null', json.dumps(payload, separators=(',', ':')))
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    print('wrote {} ({:.1f} KB)'.format(args.out, osp.getsize(args.out) / 1024))


if __name__ == '__main__':
    main()
