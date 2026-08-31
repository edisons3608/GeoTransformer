r"""Self-pair registration benchmark for a directory of meshes (e.g. talus STLs).

For every mesh we build a synthetic registration pair from the mesh itself
("self-pair"): two independent surface samplings of the same shape, a random
rigid transform applied to the source, and Gaussian jitter added to both --
the same protocol GeoTransformer uses on ModelNet40 (`ModelNetPairDataset`).
The pre-trained GeoTransformer weights are then run on each pair and the
estimated transform is compared with the ground truth.

Run one model per process (both experiment dirs contain a `config.py` /
`model.py`, so they cannot be imported into the same interpreter).

Example:
    python tools/selfpair_eval.py --model modelnet --data_dir ../talus_small
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
import trimesh
from scipy.spatial.transform import Rotation

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

EXPERIMENTS = {
    '3dmatch': 'geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn',
    'modelnet': 'geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn',
}
DEFAULT_WEIGHTS = {
    '3dmatch': 'weights/geotransformer-3dmatch.pth.tar',
    'modelnet': 'weights/geotransformer-modelnet.pth.tar',
}
# Point budget matching what each model saw during training.
DEFAULT_NUM_POINTS = {'3dmatch': 20000, 'modelnet': 1024}


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--weights', default=None, help='checkpoint (default: weights/geotransformer-MODEL.pth.tar)')
    parser.add_argument('--data_dir', default=r'C:\Users\esun3\Documents\talus_small')
    parser.add_argument('--pattern', default='*.stl')
    parser.add_argument('--num_points', type=int, default=None, help='surface samples per point cloud')
    parser.add_argument('--num_trials', type=int, default=3, help='random pairs per mesh')
    parser.add_argument('--rotation_mode', default='euler', choices=['euler', 'so3'],
                        help='euler: ModelNet-style per-axis angles in [0, magnitude]; so3: uniform random rotation')
    parser.add_argument('--rotation_magnitude', type=float, default=45.0, help='degrees (euler mode)')
    parser.add_argument('--translation_magnitude', type=float, default=0.5, help='in normalized units')
    parser.add_argument('--noise_sigma', type=float, default=0.01, help='jitter std in normalized units')
    parser.add_argument('--noise_clip', type=float, default=0.05, help='jitter clip in normalized units')
    parser.add_argument('--keep_ratio', type=float, default=None, help='plane crop keep ratio (None: full overlap)')
    parser.add_argument('--src_dir', default=None,
                        help='sample the transformed side from the matching mesh in this directory '
                             '(e.g. painted-region patches); the reference stays the full dense mesh')
    parser.add_argument('--noise_sides', default='both', choices=['both', 'src'],
                        help='jitter both clouds, or only the transformed one')
    parser.add_argument('--scale', type=float, default=1.0, help='multiplier applied after unit-sphere normalization')
    parser.add_argument('--seed', type=int, default=7351)
    parser.add_argument('--tag', default=None, help='name for the output files')
    parser.add_argument('--output_dir', default=osp.join(REPO_DIR, 'output', 'selfpair'))
    return parser


def load_pair_meshes(path, src_dir=None):
    r"""The full mesh, plus its region patch when one is supplied."""
    mesh = trimesh.load(path, process=False)
    src_mesh = None
    if src_dir:
        src_path = osp.join(src_dir, osp.basename(path))
        if not osp.exists(src_path):
            raise RuntimeError('no matching source mesh at ' + src_path)
        src_mesh = trimesh.load(src_path, process=False)
    return mesh, src_mesh


def normalize_frame(mesh, seed, num_probe=100000):
    r"""Center / radius of the mesh surface, used to put every cloud in a unit sphere."""
    probe, _ = trimesh.sample.sample_surface(mesh, num_probe, seed=seed)
    center = np.asarray(probe).mean(axis=0)
    radius = np.linalg.norm(np.asarray(probe) - center, axis=1).max()
    return center, radius


def sample_cloud(mesh, num_points, center, radius, rng):
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=int(rng.integers(1 << 31)))
    return (np.asarray(points) - center) / radius


def random_transform(rng, mode, rotation_magnitude, translation_magnitude):
    if mode == 'euler':
        euler = rng.random(3) * np.pi * rotation_magnitude / 180.0
        rotation = Rotation.from_euler('zyx', euler).as_matrix()
    else:
        rotation = Rotation.random(rng=int(rng.integers(1 << 31))).as_matrix()
    translation = rng.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def apply_transform(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def inverse_transform(transform):
    inv = np.eye(4)
    inv[:3, :3] = transform[:3, :3].T
    inv[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inv


def jitter(points, sigma, clip, rng):
    noise = np.clip(rng.normal(scale=sigma, size=points.shape), -clip, clip)
    return points + noise


def crop_with_plane(points, keep_ratio, rng):
    r"""Same plane crop as `random_crop_point_cloud_with_plane`."""
    p_normal = rng.standard_normal(3)
    p_normal = p_normal / np.linalg.norm(p_normal)
    distances = points @ p_normal
    num_keep = int(points.shape[0] * keep_ratio)
    indices = np.argsort(-distances)[:num_keep]
    return points[indices]


def build_pair(mesh, args, rng, center, radius, src_mesh=None):
    r"""One registration pair.

    The reference is always sampled from `mesh`. When `src_mesh` is given (a
    region patch of the same shape) the transformed side is sampled from that
    patch instead, at the reference's surface density, and in the reference's
    normalization frame -- so the ground-truth transform still puts the patch
    back exactly where it belongs on the whole bone.
    """
    ref_points = sample_cloud(mesh, args.num_points, center, radius, rng)
    if src_mesh is None:
        # independent ("twice") sampling of the same surface
        src_points = sample_cloud(mesh, args.num_points, center, radius, rng)
    else:
        num_src = max(512, int(round(args.num_points * src_mesh.area / mesh.area)))
        src_points = sample_cloud(src_mesh, num_src, center, radius, rng)

    transform = random_transform(rng, args.rotation_mode, args.rotation_magnitude, args.translation_magnitude)
    src_points = apply_transform(src_points, inverse_transform(transform))  # so that T maps src -> ref

    if args.keep_ratio is not None:
        ref_points = crop_with_plane(ref_points, args.keep_ratio, rng)
        src_points = crop_with_plane(src_points, args.keep_ratio, rng)

    if args.noise_sigma > 0:
        if args.noise_sides == 'both':
            ref_points = jitter(ref_points, args.noise_sigma, args.noise_clip, rng)
        src_points = jitter(src_points, args.noise_sigma, args.noise_clip, rng)

    # apply the working scale of the model (the translation scales with the points)
    ref_points = ref_points * args.scale
    src_points = src_points * args.scale
    transform = transform.copy()
    transform[:3, 3] *= args.scale

    return {
        'ref_points': ref_points.astype(np.float32),
        'src_points': src_points.astype(np.float32),
        'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
        'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
        'transform': transform.astype(np.float32),
    }


def point_to_surface_icp(mesh_vertices, mesh_faces, points, init_transform,
                         max_distance=0.05, max_iterations=60, tolerance=1e-7):
    r"""ICP that minimises distance to the triangle surface, not to sampled points.

    Each iteration finds the closest point on the mesh for every source point and
    linearises the point-to-plane objective with that triangle's own normal, so
    the target is the surface itself -- no normal estimation, no dependence on how
    densely the reference happens to be sampled.
    """
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh_vertices, dtype=np.float64)),
                                     o3d.utility.Vector3iVector(np.asarray(mesh_faces, dtype=np.int32)))
    mesh.compute_triangle_normals()
    face_normals = np.asarray(mesh.triangle_normals)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    transform = np.asarray(init_transform, dtype=np.float64).copy()
    points = np.asarray(points, dtype=np.float64)
    previous = None
    for _ in range(max_iterations):
        moved = points @ transform[:3, :3].T + transform[:3, 3]
        answer = scene.compute_closest_points(o3d.core.Tensor(moved.astype(np.float32)))
        targets = answer['points'].numpy().astype(np.float64)
        normals = face_normals[answer['primitive_ids'].numpy()]

        residual = moved - targets
        distance = np.linalg.norm(residual, axis=1)
        keep = distance < max_distance
        if keep.sum() < 6:
            break

        p, q, n = moved[keep], targets[keep], normals[keep]
        # linearised point-to-plane: unknowns are a small rotation and a translation
        A = np.hstack([np.cross(p, n), n])
        b = -np.einsum('ij,ij->i', p - q, n)
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        alpha, beta, gamma = solution[:3]
        step = np.eye(4)
        step[:3, :3] = np.array([[1.0, -gamma, beta], [gamma, 1.0, -alpha], [-beta, alpha, 1.0]])
        u, _, vt = np.linalg.svd(step[:3, :3])          # re-orthonormalise the small rotation
        step[:3, :3] = u @ vt
        step[:3, 3] = solution[3:]
        transform = step @ transform

        error = float(np.mean(np.einsum('ij,ij->i', p - q, n) ** 2))
        if previous is not None and abs(previous - error) < tolerance:
            break
        previous = error
    return transform


def ransac_icp(ref_points, src_points, ref_corr, src_corr, ransac_distance=0.1,
               icp_distance=0.05, mesh=None):
    r"""Classical baseline: RANSAC over the predicted correspondences, then ICP.

    RANSAC re-estimates the pose from the same correspondences GeoTransformer
    produced (so it is the estimator being compared, not the matcher). ICP then
    refines it: point-to-surface against `mesh` when one is given, otherwise
    point-to-plane against the reference cloud. Returns (refined, ransac_only).

    Open3D's correspondence RANSAC takes no seed, so this stage is stochastic:
    repeat runs differ by a few tenths of a degree.
    """
    import open3d as o3d

    if len(ref_corr) < 3:
        return np.eye(4), np.eye(4)

    src_corr_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(src_corr, dtype=np.float64)))
    ref_corr_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(ref_corr, dtype=np.float64)))
    matches = o3d.utility.Vector2iVector(np.tile(np.arange(len(ref_corr))[:, None], (1, 2)))
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_corr_pcd, ref_corr_pcd, matches, ransac_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999))
    ransac_transform = np.asarray(result.transformation)

    if mesh is not None:
        refined = point_to_surface_icp(mesh[0], mesh[1], src_points, ransac_transform,
                                       max_distance=icp_distance)
        return refined, ransac_transform

    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(src_points, dtype=np.float64)))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(ref_points, dtype=np.float64)))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=icp_distance * 4, max_nn=30))
    icp = o3d.pipelines.registration.registration_icp(
        source, target, icp_distance, ransac_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    return np.asarray(icp.transformation), ransac_transform


def registration_error(gt_transform, est_transform):
    gt_r, est_r = gt_transform[:3, :3], est_transform[:3, :3]
    cos = (np.trace(est_r.T @ gt_r) - 1.0) / 2.0
    rre = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    rte = np.linalg.norm(gt_transform[:3, 3] - est_transform[:3, 3])
    return rre, rte


def sixdof_error(gt_transform, est_transform):
    r"""Signed per-axis 6-DoF error.

    The rotation part is the residual rotation `R_est^T @ R_gt` decomposed into
    intrinsic xyz Euler angles -- always a small rotation, so the angles are
    unambiguous (unlike differencing the Euler angles of two large rotations).
    The translation part is the residual `t_gt - t_est` in the reference frame.
    """
    residual_r = est_transform[:3, :3].T @ gt_transform[:3, :3]
    rot = Rotation.from_matrix(residual_r).as_euler('xyz', degrees=True)
    trans = gt_transform[:3, 3] - est_transform[:3, 3]
    return rot, trans


def sixdof_stats(records):
    r"""Per-axis MAE / RMSE / max|.| / bias for the 6 degrees of freedom."""
    stats = {}
    for key, axes in (('rot_err_deg', ('rx_deg', 'ry_deg', 'rz_deg')),
                      ('trans_err_mm', ('tx_mm', 'ty_mm', 'tz_mm'))):
        errors = np.array([r[key] for r in records])  # (N, 3), signed
        for i, axis in enumerate(axes):
            column = errors[:, i]
            stats[axis] = {
                'mae': float(np.abs(column).mean()),
                'rmse': float(np.sqrt((column ** 2).mean())),
                'max_abs': float(np.abs(column).max()),
                'bias': float(column.mean()),
            }
    return stats


def main():
    args = make_parser().parse_args()
    if args.num_points is None:
        args.num_points = DEFAULT_NUM_POINTS[args.model]
    if args.weights is None:
        args.weights = osp.join(REPO_DIR, DEFAULT_WEIGHTS[args.model])
    tag = args.tag or '{}_{}{}'.format(args.model, args.rotation_mode, int(args.rotation_magnitude))

    # the experiment dir must be on the path before `config` / `model` are imported
    exp_dir = osp.join(REPO_DIR, 'experiments', EXPERIMENTS[args.model])
    sys.path.insert(0, exp_dir)
    cfg = importlib.import_module('config').make_cfg()
    create_model = importlib.import_module('model').create_model

    from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda

    files = sorted(glob.glob(osp.join(args.data_dir, args.pattern)))
    if not files:
        raise RuntimeError('no files matching {} in {}'.format(args.pattern, args.data_dir))
    print('[{}] {} meshes x {} trials, {} points/cloud'.format(tag, len(files), args.num_trials, args.num_points))

    # build every pair up front so the run is reproducible and the neighbor
    # limits are calibrated on exactly this data
    pairs, meta = [], []
    for mesh_id, path in enumerate(files):
        mesh, src_mesh = load_pair_meshes(path, args.src_dir)
        center, radius = normalize_frame(mesh, seed=args.seed + mesh_id)
        for trial in range(args.num_trials):
            rng = np.random.default_rng([args.seed, mesh_id, trial])
            pairs.append(build_pair(mesh, args, rng, center, radius, src_mesh))
            meta.append({'file': osp.basename(path), 'trial': trial, 'radius_mm': float(radius)})

    calib = pairs[:: max(1, len(pairs) // 16)][:16]
    neighbor_limits = calibrate_neighbors_stack_mode(
        calib, registration_collate_fn_stack_mode, cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size, cfg.backbone.init_radius,
    )
    print('[{}] neighbor limits: {}'.format(tag, list(neighbor_limits)))

    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict['model'])
    model.eval()

    records = []
    for index, (data_dict, info) in enumerate(zip(pairs, meta)):
        collated = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius, neighbor_limits,
        )
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output_dict = release_cuda(model(to_cuda(collated)))
        torch.cuda.synchronize()
        duration = time.time() - start

        gt_transform = data_dict['transform'].astype(np.float64)
        est_transform = np.asarray(output_dict['estimated_transform']).astype(np.float64)
        rre, rte = registration_error(gt_transform, est_transform)
        rot_err, trans_err = sixdof_error(gt_transform, est_transform)

        src_points = np.asarray(output_dict['src_points']).astype(np.float64)
        rmse = np.linalg.norm(apply_transform(src_points, est_transform)
                              - apply_transform(src_points, gt_transform), axis=1).mean()

        # inlier ratio of the predicted correspondences under the gt transform
        ref_corr = np.asarray(output_dict['ref_corr_points']).astype(np.float64)
        src_corr = apply_transform(np.asarray(output_dict['src_corr_points']).astype(np.float64), gt_transform)
        corr_dist = np.linalg.norm(ref_corr - src_corr, axis=1)
        inlier_ratio = float((corr_dist < cfg.eval.acceptance_radius).mean()) if corr_dist.size else 0.0

        mm = info['radius_mm'] / args.scale  # model units -> millimetres
        record = dict(
            info,
            rre_deg=float(rre),
            rte_mm=float(rte * mm),
            rmse_mm=float(rmse * mm),
            rte_norm=float(rte),
            rot_err_deg=[float(x) for x in rot_err],       # signed rx, ry, rz
            trans_err_mm=[float(x * mm) for x in trans_err],  # signed tx, ty, tz
            inlier_ratio=inlier_ratio,
            num_corr=int(output_dict['corr_scores'].shape[0]),
            time_s=float(duration),
        )
        records.append(record)
        print('[{}] {:3d}/{} {:24s} t{} RRE {:7.3f} deg  RMSE {:7.3f} mm  IR {:5.3f}  nCorr {:4d}  {:5.2f}s'.format(
            tag, index + 1, len(pairs), info['file'][:24], info['trial'],
            rre, record['rmse_mm'], inlier_ratio, record['num_corr'], duration), flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    rre_all = np.array([r['rre_deg'] for r in records])
    rmse_all = np.array([r['rmse_mm'] for r in records])
    rte_all = np.array([r['rte_mm'] for r in records])
    ir_all = np.array([r['inlier_ratio'] for r in records])
    summary = {
        'tag': tag,
        'model': args.model,
        'weights': args.weights,
        'num_pairs': len(records),
        'args': {k: v for k, v in vars(args).items()},
        'neighbor_limits': [int(x) for x in neighbor_limits],
        'rre_deg': {'mean': float(rre_all.mean()), 'median': float(np.median(rre_all)), 'max': float(rre_all.max())},
        'rte_mm': {'mean': float(rte_all.mean()), 'median': float(np.median(rte_all))},
        'rmse_mm': {'mean': float(rmse_all.mean()), 'median': float(np.median(rmse_all)), 'max': float(rmse_all.max())},
        'sixdof': sixdof_stats(records),
        'inlier_ratio_mean': float(ir_all.mean()),
        'recall_1deg_1mm': float(np.mean((rre_all < 1.0) & (rte_all < 1.0))),
        'recall_5deg_2mm': float(np.mean((rre_all < 5.0) & (rte_all < 2.0))),
        'recall_rmse_1mm': float(np.mean(rmse_all < 1.0)),
        'mean_time_s': float(np.mean([r['time_s'] for r in records])),
    }
    with open(osp.join(args.output_dir, tag + '.json'), 'w') as f:
        json.dump({'summary': summary, 'records': records}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
