r"""Train GeoTransformer on SSM-generated partial pairs (region -> whole bone).

Every training item is built the same way the benchmark builds its pairs: the
reference is the full generated shape, the source is that shape's painted
landmark region, randomly rotated, translated and jittered. Pairs are drawn
fresh each epoch, so 100 shapes give unlimited poses.

    python tools/train_partial.py --epochs 60                 # from scratch
    python tools/train_partial.py --epochs 30 --init pretrained

Held-out evaluation is deliberately not part of the loop -- score with
`tools/eval_test_set.py` once training is done.
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
import torch
import torch.optim as optim

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

from selfpair_eval import (  # noqa: E402
    EXPERIMENTS, DEFAULT_WEIGHTS, build_pair, normalize_frame, load_pair_meshes,
    registration_error, make_parser as eval_parser,
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=osp.join(REPO_DIR, 'output', 'ssm_bench', 'train'))
    parser.add_argument('--pattern', default='*.stl')
    parser.add_argument('--out_dir', default=osp.join(REPO_DIR, 'output', 'ssm_bench', 'runs'))
    parser.add_argument('--tag', default=None)
    parser.add_argument('--init', default='scratch', choices=['scratch', 'pretrained'])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--num_points', type=int, default=20000, help='reference points (matches the test set)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.97, help='multiplicative decay per epoch')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--val_fraction', type=float, default=0.1, help='shapes held out of training for validation')
    parser.add_argument('--rotation_mode', default='so3', choices=['euler', 'so3'])
    parser.add_argument('--rotation_magnitude', type=float, default=180.0)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--max_steps', type=int, default=None, help='stop early (smoke tests)')
    return parser


class PartialPairs(torch.utils.data.Dataset):
    r"""Region-to-whole-bone pairs, re-randomized every epoch."""

    def __init__(self, files, pair_args, seed, epoch=0):
        self.files = files
        self.pair_args = pair_args
        self.seed = seed
        self.epoch = epoch
        self.frames = {}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        if index not in self.frames:
            mesh, src_mesh = load_pair_meshes(path, osp.join(osp.dirname(path), 'paint'))
            self.frames[index] = (mesh, src_mesh) + normalize_frame(mesh, seed=self.seed + index)
        mesh, src_mesh, center, radius = self.frames[index]
        rng = np.random.default_rng([self.seed, index, self.epoch])
        return build_pair(mesh, self.pair_args, rng, center, radius, src_mesh)


def main():
    args = make_parser().parse_args()
    tag = args.tag or 'partial_{}'.format(args.init)
    out_dir = osp.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_dir = osp.join(REPO_DIR, 'experiments', EXPERIMENTS['3dmatch'])
    sys.path.insert(0, exp_dir)
    cfg = importlib.import_module('config').make_cfg()
    create_model = importlib.import_module('model').create_model
    OverallLoss = importlib.import_module('loss').OverallLoss
    Evaluator = importlib.import_module('loss').Evaluator
    from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda

    pair_args = eval_parser().parse_args(['--model', '3dmatch'])
    pair_args.num_points = args.num_points
    pair_args.rotation_mode = args.rotation_mode
    pair_args.rotation_magnitude = args.rotation_magnitude
    pair_args.noise_sides = 'src'

    files = sorted(glob.glob(osp.join(args.train_dir, args.pattern)))
    if not files:
        raise RuntimeError('no meshes in ' + args.train_dir)
    num_val = max(1, int(round(args.val_fraction * len(files))))
    val_files, train_files = files[:num_val], files[num_val:]
    train_set = PartialPairs(train_files, pair_args, seed=args.seed)
    val_set = PartialPairs(val_files, pair_args, seed=args.seed + 5000)

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_set, registration_collate_fn_stack_mode, cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size, cfg.backbone.init_radius)
    print('train {} shapes, val {} shapes, neighbor limits {}'.format(
        len(train_files), len(val_files), list(neighbor_limits)), flush=True)

    model = create_model(cfg).cuda()
    if args.init == 'pretrained':
        state = torch.load(osp.join(REPO_DIR, DEFAULT_WEIGHTS['3dmatch']), map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
    loss_fn = OverallLoss(cfg).cuda()
    evaluator = Evaluator(cfg).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    def run_pair(data_dict, train):
        collated = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius, neighbor_limits)
        collated = to_cuda(collated)
        output = model(collated)
        losses = loss_fn(output, collated)
        if train:
            optimizer.zero_grad(set_to_none=True)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        with torch.no_grad():
            metrics = evaluator(output, collated)
        result = {k: float(v.detach()) for k, v in list(losses.items()) + list(metrics.items())}
        del output, collated, losses
        return result

    history = []
    best = {'epoch': -1, 'val_rre': float('inf')}
    step = 0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_set.set_epoch(epoch)
        order = np.random.permutation(len(train_set))
        stats = []
        for i in order:
            stats.append(run_pair(train_set[int(i)], train=True))
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        scheduler.step()

        model.eval()
        val_set.set_epoch(epoch)
        val_stats = []
        with torch.no_grad():
            for i in range(len(val_set)):
                val_stats.append(run_pair(val_set[i], train=False))

        mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
        entry = {
            'epoch': epoch,
            'lr': scheduler.get_last_lr()[0],
            'train_loss': mean(stats, 'loss'),
            'train_PIR': mean(stats, 'PIR'),
            'train_IR': mean(stats, 'IR'),
            'val_loss': mean(val_stats, 'loss'),
            'val_IR': mean(val_stats, 'IR'),
            'val_RRE': mean(val_stats, 'RRE'),
            'val_RR': mean(val_stats, 'RR'),
            'minutes': (time.time() - start) / 60,
        }
        history.append(entry)
        print('epoch {:3d}  loss {:.4f}  PIR {:.3f}  IR {:.3f}  |  val loss {:.4f}  IR {:.3f}  RRE {:7.2f}  RR {:.2f}  '
              '[{:.1f} min]'.format(epoch, entry['train_loss'], entry['train_PIR'], entry['train_IR'],
                                    entry['val_loss'], entry['val_IR'], entry['val_RRE'], entry['val_RR'],
                                    entry['minutes']), flush=True)

        torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)},
                   osp.join(out_dir, 'last.pth.tar'))
        if entry['val_RRE'] < best['val_rre']:
            best = {'epoch': epoch, 'val_rre': entry['val_RRE']}
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)},
                       osp.join(out_dir, 'best.pth.tar'))
        with open(osp.join(out_dir, 'history.json'), 'w') as f:
            json.dump({'args': vars(args), 'neighbor_limits': [int(x) for x in neighbor_limits],
                       'best': best, 'history': history}, f, indent=2)
        if args.max_steps and step >= args.max_steps:
            break

    print('\ndone in {:.1f} min; best val RRE {:.2f} deg at epoch {}'.format(
        (time.time() - start) / 60, best['val_rre'], best['epoch']))
    print('checkpoints in ' + out_dir)


if __name__ == '__main__':
    main()
