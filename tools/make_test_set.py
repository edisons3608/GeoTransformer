r"""Freeze the ten held-out test cases and their pre-trained-network baseline.

Writes a manifest naming exactly which shape, trial and seed defines each case,
plus the 6-DoF errors the released 3DMatch weights produced on them, so any
model trained later is scored on identical pairs.

    python tools/make_test_set.py
"""

import json
import os
import os.path as osp
import shutil
import sys

import numpy as np

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

SHAPES_DIR = osp.join(REPO_DIR, 'output', 'random_taluses')
BASELINE = osp.join(REPO_DIR, 'output', 'selfpair_ssm', 'partial_so3180.json')
OUT_DIR = osp.join(REPO_DIR, 'output', 'ssm_bench', 'test')

# how the ten pairs are defined -- everything else follows from these
SPEC = {
    'shape_source': 'tools/generate_random_taluses.py --num_samples 10 --seed 0 (gaussian, 80 modes, +-3 sigma)',
    'reference': 'full generated shape, 20000 surface points, no noise',
    'transformed': 'painted landmark region only, density-matched (~4000 points), rotated + translated + noised',
    'pair_seed': 7351,
    'trial': 0,
    'rotation_mode': 'so3',
    'rotation_magnitude': 180.0,
    'translation_magnitude': 0.5,
    'noise_sigma': 0.01,
    'noise_clip': 0.05,
    'noise_sides': 'src',
    'num_points': 20000,
}


def main():
    with open(BASELINE) as f:
        baseline = json.load(f)
    records = [r for r in baseline['records'] if r['trial'] == SPEC['trial']]
    assert len(records) == 10, 'expected 10 trial-0 records, found {}'.format(len(records))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(osp.join(OUT_DIR, 'paint'), exist_ok=True)
    cases = []
    for index, record in enumerate(sorted(records, key=lambda r: r['file'])):
        name = record['file']
        shutil.copy2(osp.join(SHAPES_DIR, name), osp.join(OUT_DIR, name))
        shutil.copy2(osp.join(SHAPES_DIR, 'paint', name), osp.join(OUT_DIR, 'paint', name))
        cases.append({
            'index': index,
            'file': name,
            'mesh_index': index,          # position in the sorted directory listing
            'trial': SPEC['trial'],
            'radius_mm': record['radius_mm'],
            'pretrained_3dmatch': {
                'rre_deg': record['rre_deg'], 'rte_mm': record['rte_mm'], 'rmse_mm': record['rmse_mm'],
                'rot_err_deg': record['rot_err_deg'], 'trans_err_mm': record['trans_err_mm'],
                'inlier_ratio': record['inlier_ratio'], 'num_corr': record['num_corr'],
            },
        })

    rre = np.array([c['pretrained_3dmatch']['rre_deg'] for c in cases])
    manifest = {
        'name': 'ssm-talus-partial-v1',
        'num_cases': len(cases),
        'spec': SPEC,
        'meshes': 'output/ssm_bench/test (full shapes) and output/ssm_bench/test/paint (regions)',
        'baseline_summary': {
            'model': 'released geotransformer-3dmatch.pth.tar, no fine-tuning',
            'rre_deg_p25_p50_p75': [float(x) for x in np.percentile(rre, [25, 50, 75])],
            'recall_5deg_2mm': float(np.mean([(c['pretrained_3dmatch']['rre_deg'] < 5.0) and
                                              (c['pretrained_3dmatch']['rte_mm'] < 2.0) for c in cases])),
            'mean_inlier_ratio': float(np.mean([c['pretrained_3dmatch']['inlier_ratio'] for c in cases])),
        },
        'cases': cases,
    }
    path = osp.join(REPO_DIR, 'output', 'ssm_bench', 'test_cases.json')
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print('froze {} test cases -> {}'.format(len(cases), path))
    print('meshes copied to {}'.format(OUT_DIR))
    print('baseline (pre-trained 3DMatch): RRE p25/p50/p75 = {:.2f} / {:.2f} / {:.2f} deg, recall {:.0f}%'.format(
        *manifest['baseline_summary']['rre_deg_p25_p50_p75'],
        100 * manifest['baseline_summary']['recall_5deg_2mm']))


if __name__ == '__main__':
    main()
