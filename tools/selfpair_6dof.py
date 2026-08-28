r"""Print per-axis 6-DoF error tables from the JSON files written by `selfpair_eval.py`."""

import glob
import json
import os.path as osp
import sys

import numpy as np

AXES = ('rx_deg', 'ry_deg', 'rz_deg', 'tx_mm', 'ty_mm', 'tz_mm')
COLUMNS = ('dof', 'MAE', 'RMSE', 'max|e|', 'bias', 'p50', 'p95')


def percentiles(records, key, index):
    column = np.abs(np.array([r[key] for r in records])[:, index])
    return np.percentile(column, 50), np.percentile(column, 95)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else osp.join(
        osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', 'selfpair', '*.json')
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        summary, records = data['summary'], data['records']
        if 'sixdof' not in summary:
            continue
        print('\n{}  ({} pairs, {})'.format(summary['tag'], summary['num_pairs'], summary['model']))
        rows = []
        for i, axis in enumerate(AXES):
            key = 'rot_err_deg' if axis.startswith('r') else 'trans_err_mm'
            p50, p95 = percentiles(records, key, i % 3)
            s = summary['sixdof'][axis]
            rows.append((axis, '{:.4f}'.format(s['mae']), '{:.4f}'.format(s['rmse']),
                         '{:.4f}'.format(s['max_abs']), '{:+.4f}'.format(s['bias']),
                         '{:.4f}'.format(p50), '{:.4f}'.format(p95)))
        widths = [max(len(COLUMNS[i]), max(len(r[i]) for r in rows)) for i in range(len(COLUMNS))]
        fmt = '  '.join('{:>' + str(w) + '}' for w in widths)
        print(fmt.format(*COLUMNS))
        print('  '.join('-' * w for w in widths))
        for row in rows:
            print(fmt.format(*row))
    print('\nRotation: residual R_est^T R_gt as intrinsic xyz Euler angles (degrees).')
    print('Translation: t_gt - t_est per axis in the reference frame (millimetres).')


if __name__ == '__main__':
    main()
