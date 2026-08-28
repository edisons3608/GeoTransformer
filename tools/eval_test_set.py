r"""Score a checkpoint on the ten frozen test cases and print 6-DoF percentiles.

The pairs come from output/ssm_bench/test_cases.json, so every model is measured
on identical data -- the same shapes, poses and noise the released weights saw.

    python tools/eval_test_set.py --weights output/ssm_bench/runs/scratch/best.pth.tar
    python tools/eval_test_set.py --weights weights/geotransformer-3dmatch.pth.tar --tag pretrained
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

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

from selfpair_eval import (  # noqa: E402
    EXPERIMENTS, DEFAULT_WEIGHTS, build_pair, normalize_frame, load_pair_meshes, apply_transform,
    registration_error, sixdof_error, make_parser as eval_parser,
)

TEST_DIR = osp.join(REPO_DIR, 'output', 'ssm_bench', 'test')
MANIFEST = osp.join(REPO_DIR, 'output', 'ssm_bench', 'test_cases.json')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=osp.join(REPO_DIR, DEFAULT_WEIGHTS['3dmatch']))
    parser.add_argument('--tag', default=None)
    parser.add_argument('--neighbor_limits', type=int, nargs='+', default=[48, 29, 32, 37],
                        help='must match what the benchmark used, so results stay comparable')
    parser.add_argument('--out_dir', default=osp.join(REPO_DIR, 'output', 'ssm_bench', 'results'))
    return parser


def percentile_table(records, title):
    lines = ['', '{}  (n={})'.format(title, len(records)),
             '{:<9}{:>9}{:>9}{:>9}{:>9}'.format('dof', 'p25', 'p50', 'p75', 'max')]
    for i, name in enumerate(('rx_deg', 'ry_deg', 'rz_deg')):
        v = np.abs([r['rot_err_deg'][i] for r in records])
        lines.append('{:<9}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}'.format(name, *np.percentile(v, [25, 50, 75]), v.max()))
    for i, name in enumerate(('tx_mm', 'ty_mm', 'tz_mm')):
        v = np.abs([r['trans_err_mm'][i] for r in records])
        lines.append('{:<9}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}'.format(name, *np.percentile(v, [25, 50, 75]), v.max()))
    for key, name in (('rre_deg', 'RRE_deg'), ('rte_mm', 'RTE_mm'), ('rmse_mm', 'RMSE_mm')):
        v = np.array([r[key] for r in records])
        lines.append('{:<9}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}'.format(name, *np.percentile(v, [25, 50, 75]), v.max()))
    v = np.array([r['inlier_ratio'] for r in records])
    lines.append('{:<9}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}'.format('inlier', *np.percentile(v, [25, 50, 75]), v.max()))
    return '\n'.join(lines)


def main():
    args = make_parser().parse_args()
    tag = args.tag or osp.basename(osp.dirname(args.weights))

    with open(MANIFEST) as f:
        manifest = json.load(f)
    spec = manifest['spec']

    exp_dir = osp.join(REPO_DIR, 'experiments', EXPERIMENTS['3dmatch'])
    sys.path.insert(0, exp_dir)
    cfg = importlib.import_module('config').make_cfg()
    create_model = importlib.import_module('model').create_model
    from geotransformer.utils.data import registration_collate_fn_stack_mode as collate
    from geotransformer.utils.torch import to_cuda, release_cuda

    pair_args = eval_parser().parse_args(['--model', '3dmatch'])
    pair_args.num_points = spec['num_points']
    pair_args.rotation_mode = spec['rotation_mode']
    pair_args.rotation_magnitude = spec['rotation_magnitude']
    pair_args.noise_sides = spec['noise_sides']
    pair_args.seed = spec['pair_seed']

    model = create_model(cfg).cuda()
    state = torch.load(args.weights, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()

    files = sorted(glob.glob(osp.join(TEST_DIR, '*.stl')))
    assert len(files) == manifest['num_cases'], 'test set is {} meshes, manifest says {}'.format(
        len(files), manifest['num_cases'])

    records = []
    for case in manifest['cases']:
        path = osp.join(TEST_DIR, case['file'])
        mesh, src_mesh = load_pair_meshes(path, osp.join(TEST_DIR, 'paint'))
        center, radius = normalize_frame(mesh, seed=pair_args.seed + case['mesh_index'])
        rng = np.random.default_rng([pair_args.seed, case['mesh_index'], case['trial']])
        data_dict = build_pair(mesh, pair_args, rng, center, radius, src_mesh)

        with torch.no_grad():
            output = release_cuda(model(to_cuda(collate(
                [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
                cfg.backbone.init_radius, args.neighbor_limits))))

        gt = data_dict['transform'].astype(np.float64)
        est = np.asarray(output['estimated_transform']).astype(np.float64)
        rre, rte = registration_error(gt, est)
        rot_err, trans_err = sixdof_error(gt, est)
        src_points = np.asarray(output['src_points'], dtype=np.float64)
        rmse = np.linalg.norm(apply_transform(src_points, est) - apply_transform(src_points, gt), axis=1).mean()
        ref_corr = np.asarray(output['ref_corr_points'], dtype=np.float64)
        src_corr = apply_transform(np.asarray(output['src_corr_points'], dtype=np.float64), gt)
        ir = float((np.linalg.norm(ref_corr - src_corr, axis=1) < cfg.eval.acceptance_radius).mean())

        records.append({
            'file': case['file'], 'trial': case['trial'], 'radius_mm': radius,
            'rre_deg': float(rre), 'rte_mm': float(rte * radius), 'rmse_mm': float(rmse * radius),
            'rot_err_deg': [float(x) for x in rot_err],
            'trans_err_mm': [float(x * radius) for x in trans_err],
            'inlier_ratio': ir, 'num_corr': int(output['corr_scores'].shape[0]),
        })
        print('{:<24} RRE {:8.2f} deg  RTE {:7.3f} mm  IR {:.3f}'.format(
            case['file'].replace('.stl', ''), records[-1]['rre_deg'], records[-1]['rte_mm'], ir), flush=True)

    rre = np.array([r['rre_deg'] for r in records])
    rte = np.array([r['rte_mm'] for r in records])
    summary = {
        'tag': tag, 'weights': args.weights, 'checkpoint_epoch': state.get('epoch'),
        'recall_5deg_2mm': float(np.mean((rre < 5.0) & (rte < 2.0))),
        'recall_10deg_5mm': float(np.mean((rre < 10.0) & (rte < 5.0))),
        'rre_deg_p25_p50_p75': [float(x) for x in np.percentile(rre, [25, 50, 75])],
        'mean_inlier_ratio': float(np.mean([r['inlier_ratio'] for r in records])),
    }
    print(percentile_table(records, 'test set: {}'.format(tag)))
    print('\nrecall  <5deg/2mm {:.0f}%   <10deg/5mm {:.0f}%   mean inlier ratio {:.3f}'.format(
        100 * summary['recall_5deg_2mm'], 100 * summary['recall_10deg_5mm'], summary['mean_inlier_ratio']))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, tag + '.json'), 'w') as f:
        json.dump({'summary': summary, 'records': records}, f, indent=2)


if __name__ == '__main__':
    main()
