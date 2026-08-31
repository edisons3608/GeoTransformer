r"""Train GeoTransformer to register sampled landmark points to the landmark regions.

Same pairs as `train_landmarks.py` on the transformed side -- `--points_per_region`
points sampled from each of the six regions, rotated, translated and jittered --
but the reference is no longer the whole bone: it is the 462 mesh vertices that
lie inside those same six regions. Both clouds therefore cover the same small
patches of anatomy instead of a handful of points against a whole surface.

    python tools/train_landmark_regions.py --epochs 30            # fine-tune (default)
    python tools/train_landmark_regions.py --init scratch --epochs 60
    python tools/train_landmark_regions.py --eval --weights <ckpt>

For a given seed the transformed cloud and the ground-truth transform are
bit-identical to the whole-bone variant, so the two runs are directly comparable.

This is a thin entry point: the pair construction and training loop live in
`train_landmarks.py`, which this calls with `--reference regions`.
"""

import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import train_landmarks  # noqa: E402


def main():
    argv = sys.argv[1:]
    if '--reference' in argv:
        raise SystemExit('this script is the "regions" reference; use train_landmarks.py to choose another')
    sys.argv = [sys.argv[0]] + argv + ['--reference', 'regions']
    train_landmarks.main()


if __name__ == '__main__':
    main()
