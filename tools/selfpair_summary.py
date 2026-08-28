r"""Print a comparison table of the JSON files written by `selfpair_eval.py`."""

import glob
import json
import os.path as osp
import sys

import numpy as np

HEADER = ('run', 'pairs', 'RRE med', 'RRE mean', 'RMSE med', 'RMSE mean', 'IR', 'RR<5deg/2mm', 'RR<1deg/1mm', 's/pair')


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else osp.join(
        osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', 'selfpair', '*.json')
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        s = data['summary']
        rows.append((
            s['tag'], str(s['num_pairs']),
            '{:.2f}'.format(s['rre_deg']['median']), '{:.2f}'.format(s['rre_deg']['mean']),
            '{:.3f}'.format(s['rmse_mm']['median']), '{:.3f}'.format(s['rmse_mm']['mean']),
            '{:.3f}'.format(s['inlier_ratio_mean']),
            '{:.1f}%'.format(100 * s['recall_5deg_2mm']), '{:.1f}%'.format(100 * s['recall_1deg_1mm']),
            '{:.2f}'.format(s['mean_time_s']),
        ))
    if not rows:
        print('no results found for ' + pattern)
        return
    widths = [max(len(HEADER[i]), max(len(r[i]) for r in rows)) for i in range(len(HEADER))]
    fmt = '  '.join('{:<' + str(w) + '}' for w in widths)
    print(fmt.format(*HEADER))
    print('  '.join('-' * w for w in widths))
    for row in rows:
        print(fmt.format(*row))
    print('\nRRE in degrees, RMSE/RTE in millimetres, IR = inlier ratio of predicted correspondences.')


if __name__ == '__main__':
    main()
