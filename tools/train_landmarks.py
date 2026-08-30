r"""Train GeoTransformer to register a handful of landmark points to a whole bone.

The transformed cloud is not a surface patch: it is `--points_per_region` points
sampled from each of the six landmark regions in the model file (4 x 6 = 24 points
by default), rotated, translated and jittered. The reference stays the full dense
shape. Shapes are drawn from the SSM on the fly, so train / val / test are disjoint
draws of the same generator.

    python tools/train_landmarks.py --epochs 30                    # fine-tune (default)
    python tools/train_landmarks.py --init scratch --epochs 60
    python tools/train_landmarks.py --eval --weights <ckpt>        # score a checkpoint

Test shapes use seeds disjoint from training, and the pair for a given shape is
fully determined by its seed, so evaluation is repeatable.
"""

import argparse
import importlib
import json
import os
import os.path as osp
import sys
import time

import h5py
import numpy as np
import torch
import torch.optim as optim
import trimesh

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

from selfpair_eval import (  # noqa: E402
    EXPERIMENTS, DEFAULT_WEIGHTS, random_transform, inverse_transform, apply_transform,
    jitter, registration_error, sixdof_error,
)

DEFAULT_MODEL = r'C:\Users\esun3\OneDrive - Stryker\Documents\tal_left_reg_6_loc.h5'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', default=DEFAULT_MODEL, help='SSM + 6 landmark regions')
    parser.add_argument('--out_dir', default=osp.join(REPO_DIR, 'output', 'landmark_bench'))
    parser.add_argument('--tag', default=None)
    parser.add_argument('--init', default='pretrained', choices=['pretrained', 'scratch'])
    parser.add_argument('--weights', default=None, help='checkpoint for --eval')
    parser.add_argument('--eval', action='store_true', help='score a checkpoint on the test shapes')
    parser.add_argument('--points_per_region', type=int, default=4)
    parser.add_argument('--points_in_patch', type=int, default=0,
                        help='points grouped per superpoint; 0 = auto from the source cloud size '
                             '(the 64 used for dense pairs exceeds the whole sparse cloud)')
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--num_val', type=int, default=10)
    parser.add_argument('--num_test', type=int, default=10)
    parser.add_argument('--num_modes', default='95%')
    parser.add_argument('--sigma_range', type=float, default=3.0)
    parser.add_argument('--num_points', type=int, default=20000, help='reference points')
    parser.add_argument('--rotation_mode', default='so3', choices=['euler', 'so3'])
    parser.add_argument('--rotation_magnitude', type=float, default=180.0)
    parser.add_argument('--translation_magnitude', type=float, default=0.5)
    parser.add_argument('--noise_sigma', type=float, default=0.01)
    parser.add_argument('--noise_clip', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--train_seed', type=int, default=101)
    parser.add_argument('--test_seed', type=int, default=909)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--resume', nargs='?', const='auto', default=None,
                        help="continue a run: 'auto' takes last.pth.tar from the run's own directory")
    return parser


class LandmarkShapes:
    r"""SSM shapes plus the six landmark regions, resolved once on the mean shape."""

    def __init__(self, model_file, num_modes_spec, points_per_region):
        with h5py.File(model_file, 'r') as f:
            self.mean = f['model/mean_shape'][:]
            self.pcs = f['model/principal_components'][:]
            self.eigenvalues = f['model/eigenvalues'][:]
            self.faces = f['topology/template_faces'][:]
            centers = f['landmarks/paint_centers'][:]
            radii = f['landmarks/paint_radii'][:]
            region_ids = f['landmarks/paint_region_ids'][:]
            self.region_keys = f['landmarks/paint_region_keys'][:]
            self.norm_scale = float(f.attrs['norm_scale'])
        self.sigmas = np.sqrt(np.maximum(self.eigenvalues, 0.0))
        self.points_per_region = points_per_region

        cumulative = np.cumsum(self.eigenvalues) / self.eigenvalues.sum()
        if isinstance(num_modes_spec, str) and num_modes_spec.endswith('%'):
            self.num_modes = int(np.searchsorted(cumulative, float(num_modes_spec.rstrip('%')) / 100.0) + 1)
        else:
            self.num_modes = int(num_modes_spec)

        # a face belongs to a region when all three of its vertices sit inside that
        # region's spheres -- resolved on the mean shape, so every sampled shape
        # inherits the same regions through vertex correspondence
        vertices = self.mean.reshape(-1, 3)
        distance = np.linalg.norm(vertices[:, None, :] - centers[None, :, :], axis=2)
        self.region_faces = []
        for key in self.region_keys:
            sel = region_ids == key
            mask = (distance[:, sel] <= radii[sel][None, :]).any(axis=1)
            self.region_faces.append(self.faces[mask[self.faces].all(axis=1)])
        if any(len(f) == 0 for f in self.region_faces):
            raise RuntimeError('a landmark region resolved to zero faces')

    def shape(self, seed):
        r"""One SSM draw: gaussian z-scores truncated to +-sigma_range, in millimetres."""
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(self.num_modes)
        while True:
            outside = np.abs(z) > 3.0
            if not outside.any():
                break
            z[outside] = rng.standard_normal(int(outside.sum()))
        vertices = (self.mean + (z * self.sigmas[:self.num_modes]) @ self.pcs[:self.num_modes]).reshape(-1, 3)
        return vertices / self.norm_scale, z

    def pair(self, seed, args):
        r"""Reference = whole shape; source = a few points per landmark region, moved."""
        vertices, _ = self.shape(seed)
        rng = np.random.default_rng(seed + 7_000_000)
        mesh = trimesh.Trimesh(vertices, self.faces, process=False)

        probe, _ = trimesh.sample.sample_surface(mesh, 100000, seed=int(seed))
        center = np.asarray(probe).mean(axis=0)
        radius = np.linalg.norm(np.asarray(probe) - center, axis=1).max()

        ref_points, _ = trimesh.sample.sample_surface(mesh, args.num_points, seed=int(rng.integers(1 << 31)))
        ref_points = (np.asarray(ref_points) - center) / radius

        picked = []
        for faces in self.region_faces:
            region = trimesh.Trimesh(vertices, faces, process=False)
            points, _ = trimesh.sample.sample_surface(region, self.points_per_region,
                                                      seed=int(rng.integers(1 << 31)))
            picked.append((np.asarray(points) - center) / radius)
        src_points = np.concatenate(picked, axis=0)

        transform = random_transform(rng, args.rotation_mode, args.rotation_magnitude, args.translation_magnitude)
        src_points = apply_transform(src_points, inverse_transform(transform))
        if args.noise_sigma > 0:
            src_points = jitter(src_points, args.noise_sigma, args.noise_clip, rng)

        return {
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((len(ref_points), 1), dtype=np.float32),
            'src_feats': np.ones((len(src_points), 1), dtype=np.float32),
            'transform': transform.astype(np.float32),
        }, radius


def percentile_table(records, title):
    lines = ['', '{}  (n={})'.format(title, len(records)),
             '{:<9}{:>9}{:>9}{:>9}{:>9}'.format('dof', 'p25', 'p50', 'p75', 'max')]
    for i, name in enumerate(('rx_deg', 'ry_deg', 'rz_deg')):
        v = np.abs([r['rot_err_deg'][i] for r in records])
        lines.append('{:<9}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}'.format(name, *np.percentile(v, [25, 50, 75]), v.max()))
    for i, name in enumerate(('tx_mm', 'ty_mm', 'tz_mm')):
        v = np.abs([r['trans_err_mm'][i] for r in records])
        lines.append('{:<9}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}'.format(name, *np.percentile(v, [25, 50, 75]), v.max()))
    for key, name in (('rre_deg', 'RRE_deg'), ('rte_mm', 'RTE_mm')):
        v = np.array([r[key] for r in records])
        lines.append('{:<9}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}'.format(name, *np.percentile(v, [25, 50, 75]), v.max()))
    v = np.array([r['inlier_ratio'] for r in records])
    lines.append('{:<9}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}'.format('inlier', *np.percentile(v, [25, 50, 75]), v.max()))
    return '\n'.join(lines)


def main():
    args = make_parser().parse_args()
    tag = args.tag or ('landmarks_{}pts_{}'.format(6 * args.points_per_region, args.init))
    out_dir = osp.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)

    shapes = LandmarkShapes(args.model_file, args.num_modes, args.points_per_region)
    print('model {}\n{} modes, {} landmark regions, {} points per region -> {} source points'.format(
        osp.basename(args.model_file), shapes.num_modes, len(shapes.region_keys),
        args.points_per_region, 6 * args.points_per_region), flush=True)

    exp_dir = osp.join(REPO_DIR, 'experiments', EXPERIMENTS['3dmatch'])
    sys.path.insert(0, exp_dir)
    cfg = importlib.import_module('config').make_cfg()
    num_src = 6 * args.points_per_region
    cfg.model.num_points_in_patch = args.points_in_patch or max(4, min(64, num_src // 2))
    print('points per superpoint patch: {} (source cloud is {} points)'.format(
        cfg.model.num_points_in_patch, num_src), flush=True)
    create_model = importlib.import_module('model').create_model
    loss_module = importlib.import_module('loss')
    from geotransformer.utils.data import registration_collate_fn_stack_mode as collate
    from geotransformer.utils.data import calibrate_neighbors_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda

    train_seeds = [args.train_seed + i for i in range(args.num_train)]
    val_seeds = [args.train_seed + 500_000 + i for i in range(args.num_val)]
    test_seeds = [args.test_seed + i for i in range(args.num_test)]

    class Calib(torch.utils.data.Dataset):
        def __len__(self):
            return min(16, len(train_seeds))

        def __getitem__(self, index):
            return shapes.pair(train_seeds[index], args)[0]

    neighbor_limits = calibrate_neighbors_stack_mode(
        Calib(), collate, cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius)
    print('neighbor limits {}'.format(list(neighbor_limits)), flush=True)

    model = create_model(cfg).cuda()
    if args.eval:
        state = torch.load(args.weights, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
    elif args.init == 'pretrained':
        state = torch.load(osp.join(REPO_DIR, DEFAULT_WEIGHTS['3dmatch']), map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])

    def evaluate(seeds, label):
        model.eval()
        records = []
        for seed in seeds:
            data_dict, radius = shapes.pair(seed, args)
            with torch.no_grad():
                output = release_cuda(model(to_cuda(collate(
                    [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
                    cfg.backbone.init_radius, neighbor_limits))))
            gt = data_dict['transform'].astype(np.float64)
            est = np.asarray(output['estimated_transform']).astype(np.float64)
            rre, rte = registration_error(gt, est)
            rot, trans = sixdof_error(gt, est)
            ref_corr = np.asarray(output['ref_corr_points'], dtype=np.float64)
            src_corr = apply_transform(np.asarray(output['src_corr_points'], dtype=np.float64), gt)
            ir = float((np.linalg.norm(ref_corr - src_corr, axis=1) < cfg.eval.acceptance_radius).mean()) \
                if len(ref_corr) else 0.0
            records.append({'seed': int(seed), 'rre_deg': float(rre), 'rte_mm': float(rte * radius),
                            'rot_err_deg': [float(x) for x in rot],
                            'trans_err_mm': [float(x * radius) for x in trans],
                            'inlier_ratio': ir, 'num_corr': int(output['corr_scores'].shape[0])})
        rre = np.array([r['rre_deg'] for r in records])
        rte = np.array([r['rte_mm'] for r in records])
        summary = {'recall_5deg_2mm': float(np.mean((rre < 5) & (rte < 2))),
                   'recall_10deg_5mm': float(np.mean((rre < 10) & (rte < 5))),
                   'rre_p50': float(np.median(rre)),
                   'mean_inlier_ratio': float(np.mean([r['inlier_ratio'] for r in records]))}
        print(percentile_table(records, label))
        print('\nrecall <5deg/2mm {:.0f}%   <10deg/5mm {:.0f}%   mean inlier ratio {:.3f}'.format(
            100 * summary['recall_5deg_2mm'], 100 * summary['recall_10deg_5mm'], summary['mean_inlier_ratio']),
            flush=True)
        return records, summary

    if args.eval:
        records, summary = evaluate(test_seeds, 'test shapes: ' + tag)
        with open(osp.join(out_dir, 'test_results.json'), 'w') as f:
            json.dump({'summary': summary, 'weights': args.weights, 'records': records}, f, indent=2)
        return

    loss_fn = loss_module.OverallLoss(cfg).cuda()
    evaluator = loss_module.Evaluator(cfg).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    start_epoch = 0
    history, best = [], {'epoch': -1, 'val_rre': float('inf')}
    if args.resume:
        resume_path = osp.join(out_dir, 'last.pth.tar') if args.resume == 'auto' else args.resume
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('checkpoint has no optimizer state -- Adam moments restart', flush=True)
        start_epoch = checkpoint['epoch'] + 1
        for _ in range(start_epoch):
            scheduler.step()          # put the decayed lr back where it was
        history_path = osp.join(out_dir, 'history.json')
        if osp.exists(history_path):
            with open(history_path) as f:
                previous = json.load(f)
            history = [e for e in previous['history'] if e['epoch'] < start_epoch]
            best = previous['best']
        print('resuming {} at epoch {} (lr {:.2e}, best val RRE {:.2f})'.format(
            osp.basename(resume_path), start_epoch, scheduler.get_last_lr()[0], best['val_rre']), flush=True)

    def run(data_dict, train):
        collated = to_cuda(collate([data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
                                   cfg.backbone.init_radius, neighbor_limits))
        output = model(collated)
        losses = loss_fn(output, collated)
        if train:
            optimizer.zero_grad(set_to_none=True)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        with torch.no_grad():
            metrics = evaluator(output, collated)
        return {k: float(v.detach()) for k, v in list(losses.items()) + list(metrics.items())}

    step = 0
    elapsed_before = history[-1]['minutes'] if history else 0.0
    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        stats = []
        for i in np.random.permutation(len(train_seeds)):
            stats.append(run(shapes.pair(train_seeds[int(i)] + epoch * 1_000_000, args)[0], train=True))
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_stats = [run(shapes.pair(seed, args)[0], train=False) for seed in val_seeds]

        mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
        entry = {'epoch': epoch, 'lr': scheduler.get_last_lr()[0],
                 'train_loss': mean(stats, 'loss'), 'train_IR': mean(stats, 'IR'), 'train_PIR': mean(stats, 'PIR'),
                 'val_loss': mean(val_stats, 'loss'), 'val_IR': mean(val_stats, 'IR'),
                 'val_RRE': mean(val_stats, 'RRE'), 'val_RR': mean(val_stats, 'RR'),
                 'minutes': elapsed_before + (time.time() - start) / 60}
        history.append(entry)
        print('epoch {:3d}  loss {:.4f}  PIR {:.3f}  IR {:.3f}  |  val loss {:.4f}  IR {:.3f}  RRE {:7.2f}  '
              'RR {:.2f}  [{:.1f} min]'.format(epoch, entry['train_loss'], entry['train_PIR'], entry['train_IR'],
                                               entry['val_loss'], entry['val_IR'], entry['val_RRE'],
                                               entry['val_RR'], entry['minutes']), flush=True)

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'args': vars(args)}, osp.join(out_dir, 'last.pth.tar'))
        if entry['val_RRE'] < best['val_rre']:
            best = {'epoch': epoch, 'val_rre': entry['val_RRE']}
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)},
                       osp.join(out_dir, 'best.pth.tar'))
        with open(osp.join(out_dir, 'history.json'), 'w') as f:
            json.dump({'args': vars(args), 'neighbor_limits': [int(x) for x in neighbor_limits],
                       'best': best, 'history': history}, f, indent=2)
        if args.max_steps and step >= args.max_steps:
            break

    print('\ntrained in {:.1f} min; best val RRE {:.2f} deg at epoch {}'.format(
        (time.time() - start) / 60, best['val_rre'], best['epoch']), flush=True)

    state = torch.load(osp.join(out_dir, 'best.pth.tar'), map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    records, summary = evaluate(test_seeds, 'test shapes: ' + tag)
    with open(osp.join(out_dir, 'test_results.json'), 'w') as f:
        json.dump({'summary': summary, 'epoch': best['epoch'], 'records': records}, f, indent=2)
    print('checkpoints and results in ' + out_dir)


if __name__ == '__main__':
    main()
