r"""Generate random taluses from a statistical shape model.

Draws a random shape coefficient for every retained mode in [-3 sigma, +3 sigma]
(sigma_i = sqrt(eigenvalue_i)) and reconstructs

    shape = mean_shape + sum_i c_i * sigma_i * pc_i

writing one mesh per sample plus a CSV of the coefficients that produced them.

    python tools/generate_random_taluses.py --num_samples 10

The model file stores unit-norm principal components, eigenvalues as coefficient
variances, and a centroid-size-normalized mean shape; `norm_scale` in the file
attributes converts the normalized coordinates back to millimetres.
"""

import argparse
import csv
import os
import os.path as osp

import h5py
import numpy as np
import trimesh

DEFAULT_MODEL = r'C:\Users\esun3\OneDrive - Stryker\Documents\tal_left_reg.h5'
DEFAULT_OUT = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', 'random_taluses')


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=DEFAULT_MODEL, help='shape model .h5')
    parser.add_argument('--out_dir', default=DEFAULT_OUT)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_modes', default='95%',
                        help="modes to sample: an integer, a variance target like '95%%', or 'all'")
    parser.add_argument('--sigma_range', type=float, default=3.0, help='coefficients bounded to [-R, +R] sigma')
    parser.add_argument('--distribution', default='gaussian', choices=['gaussian', 'uniform'],
                        help='normal truncated to the range (default), or uniform over it')
    parser.add_argument('--format', default='stl', choices=['stl', 'ply', 'obj'])
    parser.add_argument('--units', default='mm', choices=['mm', 'normalized'],
                        help='mm undoes the centroid-size normalization using the model norm_scale')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_npz', action='store_true', help='also save all vertices in one .npz')
    parser.add_argument('--preview', action='store_true', help='also write a contact sheet PNG of the samples')
    parser.add_argument('--export_paint', action='store_true',
                        help="also write each sample's painted region as a patch mesh in <out_dir>/paint")
    return parser


def load_model(path):
    with h5py.File(path, 'r') as f:
        model = {
            'mean': f['model/mean_shape'][:],
            'pcs': f['model/principal_components'][:],
            'eigenvalues': f['model/eigenvalues'][:],
            'faces': f['topology/template_faces'][:],
            'num_training': f['model/corresponded_pc'].shape[0],
            'attrs': dict(f.attrs),
            'paint_centers': f['landmarks/paint_centers'][:],
            'paint_radii': f['landmarks/paint_radii'][:],
        }
    model['sigmas'] = np.sqrt(np.maximum(model['eigenvalues'], 0.0))
    return model


def painted_faces(model):
    r"""Faces of the painted region, found on the mean shape.

    The paint is stored as spheres in the model's own (normalized) frame. Every
    generated shape shares the template's vertex indexing, so the mask found here
    carries over to each sample unchanged -- that correspondence is the whole
    point of doing it on the mean shape rather than per sample.
    """
    vertices = model['mean'].reshape(-1, 3)
    centers, radii = model['paint_centers'], model['paint_radii']
    distances = np.linalg.norm(vertices[:, None, :] - centers[None, :, :], axis=2)
    vertex_mask = (distances <= radii[None, :]).any(axis=1)
    faces = model['faces']
    return faces[vertex_mask[faces].all(axis=1)], vertex_mask


def resolve_num_modes(spec, eigenvalues):
    r"""An integer, a variance target ('95%'), or 'all'."""
    total = len(eigenvalues)
    if spec == 'all':
        return total
    if isinstance(spec, str) and spec.strip().endswith('%'):
        target = float(spec.strip().rstrip('%')) / 100.0
        cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
        return int(np.searchsorted(cumulative, target) + 1)
    return min(int(spec), total)


def sample_coefficients(rng, num_samples, num_modes, sigma_range, distribution):
    r"""Coefficients in units of sigma, one row per generated shape."""
    if distribution == 'uniform':
        return rng.uniform(-sigma_range, sigma_range, size=(num_samples, num_modes))
    coefficients = rng.standard_normal((num_samples, num_modes))
    while True:  # resample the tails instead of clipping, which would pile mass on the bounds
        outside = np.abs(coefficients) > sigma_range
        if not outside.any():
            return coefficients
        coefficients[outside] = rng.standard_normal(int(outside.sum()))


def write_preview(vertices, faces, mean_vertices, labels, path):
    r"""Contact sheet of the generated shapes, with the mean shape first for reference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shapes = [mean_vertices] + list(vertices)
    titles = ['mean shape'] + list(labels)
    cols = min(6, len(shapes))
    rows = int(np.ceil(len(shapes) / cols))
    fig = plt.figure(figsize=(2.35 * cols, 2.5 * rows), facecolor='#fcfcfb')
    span = np.abs(np.concatenate(shapes) - np.concatenate(shapes).mean(axis=0)).max()
    for i, (verts, title) in enumerate(zip(shapes, titles)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        center = verts.mean(axis=0)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color='#c8d3de' if i else '#eb6834', edgecolor='none',
                        shade=True, linewidth=0)
        ax.set_title(title, fontsize=8, color='#52514e', pad=0)
        for setter, c in ((ax.set_xlim, center[0]), (ax.set_ylim, center[1]), (ax.set_zlim, center[2])):
            setter(c - span, c + span)
        ax.set_axis_off()
        ax.view_init(elev=18, azim=35)
        ax.set_box_aspect((1, 1, 1), zoom=1.4)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0, wspace=0, hspace=0.05)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = make_parser().parse_args()
    model = load_model(args.model)
    eigenvalues, sigmas, pcs = model['eigenvalues'], model['sigmas'], model['pcs']

    num_modes = resolve_num_modes(args.num_modes, eigenvalues)
    explained = eigenvalues[:num_modes].sum() / eigenvalues.sum()
    scale = 1.0 / model['attrs']['norm_scale'] if args.units == 'mm' else 1.0

    print('model      : {}'.format(args.model))
    print('training set: {} shapes, {} modes, {} vertices'.format(
        model['num_training'], len(eigenvalues), model['attrs']['n_points']))
    print('sampling   : {} modes ({:.1f}% of shape variance), {} in [-{:g}, +{:g}] sigma'.format(
        num_modes, 100 * explained, args.distribution, args.sigma_range, args.sigma_range))

    rng = np.random.default_rng(args.seed)
    coefficients = sample_coefficients(rng, args.num_samples, num_modes, args.sigma_range, args.distribution)

    # shape = mean + sum_i (c_i * sigma_i) * pc_i, all samples at once
    deltas = (coefficients * sigmas[:num_modes]) @ pcs[:num_modes]
    vertices = (model['mean'] + deltas).reshape(args.num_samples, -1, 3) * scale
    faces = model['faces']

    os.makedirs(args.out_dir, exist_ok=True)
    paint_dir = osp.join(args.out_dir, 'paint')
    if args.export_paint:
        paint_faces, vertex_mask = painted_faces(model)
        os.makedirs(paint_dir, exist_ok=True)
        patch_area = trimesh.Trimesh(vertices[0], paint_faces, process=False).area
        full_area = trimesh.Trimesh(vertices[0], faces, process=False).area
        print('painted region: {} of {} vertices, {} faces, {:.0f} mm2 ({:.1f}% of the surface)'.format(
            int(vertex_mask.sum()), len(vertex_mask), len(paint_faces), patch_area, 100 * patch_area / full_area))

    rows = []
    for i in range(args.num_samples):
        mesh = trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False)
        name = 'talus_random_{:03d}.{}'.format(i, args.format)
        mesh.export(osp.join(args.out_dir, name))
        if args.export_paint:
            patch = trimesh.Trimesh(vertices=vertices[i], faces=paint_faces, process=False)
            patch.remove_unreferenced_vertices()
            patch.export(osp.join(paint_dir, name))
        # Mahalanobis distance in shape space: how unusual this draw is overall
        mahalanobis = float(np.linalg.norm(coefficients[i]))
        extent = mesh.extents
        rows.append([name, '{:.3f}'.format(mahalanobis), '{:.1f}'.format(extent[0]),
                     '{:.1f}'.format(extent[1]), '{:.1f}'.format(extent[2]),
                     '{:.1f}'.format(mesh.volume if mesh.is_volume else float('nan'))]
                    + ['{:.4f}'.format(c) for c in coefficients[i]])
        print('{}  extent {:5.1f} x {:5.1f} x {:5.1f} {}  |c| = {:.2f}'.format(
            name, extent[0], extent[1], extent[2], args.units, mahalanobis))

    header = (['file', 'mahalanobis', 'extent_x', 'extent_y', 'extent_z', 'volume']
              + ['c{:03d}'.format(k) for k in range(num_modes)])
    csv_path = osp.join(args.out_dir, 'coefficients.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    if args.preview:
        preview_path = osp.join(args.out_dir, 'preview.png')
        write_preview(vertices, faces, model['mean'].reshape(-1, 3) * scale,
                      [r[0].split('.')[0][-3:] + '  |c| ' + r[1] for r in rows], preview_path)
        print('preview   : {}'.format(preview_path))

    if args.save_npz:
        extra = {'paint_vertex_mask': vertex_mask, 'paint_faces': paint_faces} if args.export_paint else {}
        np.savez_compressed(osp.join(args.out_dir, 'random_taluses.npz'),
                            vertices=vertices.astype(np.float32), faces=faces,
                            coefficients=coefficients, sigmas=sigmas[:num_modes], units=args.units, **extra)

    print('\nwrote {} meshes + coefficients.csv to {}'.format(args.num_samples, args.out_dir))


if __name__ == '__main__':
    main()
