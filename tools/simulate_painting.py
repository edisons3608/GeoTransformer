r"""Simulate a surgeon painting each of the six landmark regions with a tracked probe.

The model file records painting as brush spheres (`paint_centers` + `paint_radii`),
so this reproduces the same act rather than sampling points independently: a probe
is dragged across the surface inside a region, leaving a band of points behind it.

Each stroke is
  1. a walk over mesh vertices inside the region, with direction momentum so the
     path sweeps instead of jittering,
  2. resampled at the spacing the probe's rate and sweep speed imply,
  3. spread sideways within the brush radius and re-projected onto the surface,
  4. jittered by the tracker's noise.

    python tools/simulate_painting.py --seed 909 --preview
    python tools/simulate_painting.py --seed 909 --pattern uniform     # the old baseline
    python tools/simulate_painting.py --num_shapes 5 --out_dir output/painted

Import `PaintSimulator` to use the painted clouds as the transformed side of a
registration pair.
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import trimesh

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

from train_landmarks import LandmarkShapes, DEFAULT_MODEL  # noqa: E402

REGION_COLORS = np.array([
    [0.165, 0.471, 0.839], [0.922, 0.408, 0.204], [0.106, 0.686, 0.478],
    [0.929, 0.631, 0.000], [0.910, 0.482, 0.643], [0.290, 0.227, 0.655],
])


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', default=DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=909, help='SSM shape seed')
    parser.add_argument('--num_shapes', type=int, default=1, help='consecutive seeds to paint')
    parser.add_argument('--pattern', default='sweep', choices=['sweep', 'stroke', 'uniform'],
                        help='sweep: back-and-forth passes that cover the region; '
                             'stroke: unguided probe walks; uniform: independent surface samples')
    parser.add_argument('--sweep_overlap', type=float, default=1.2,
                        help='spacing between passes, in brush radii (<2 overlaps)')
    parser.add_argument('--coverage', type=float, default=0.75,
                        help='roughly what fraction of each region to paint; 1 = sweep all of it')
    parser.add_argument('--pass_smoothing', type=float, default=5.0,
                        help='mm of smoothing along a pass; higher = straighter sweeps')
    parser.add_argument('--wobble_mm', type=float, default=0.4, help='lateral drift of the probe')
    parser.add_argument('--wobble_length_mm', type=float, default=6.0,
                        help='how far the probe drifts before changing its mind')
    parser.add_argument('--strokes_per_region', type=int, default=2)
    parser.add_argument('--stroke_length_mm', type=float, default=12.0,
                        help='target path length per stroke, capped by the region size')
    parser.add_argument('--probe_rate_hz', type=float, default=200.0, help='probe sample rate')
    parser.add_argument('--probe_speed_mm_s', type=float, default=10.0, help='how fast the probe is swept')
    parser.add_argument('--spacing_mm', type=float, default=0.0,
                        help='override the spacing directly; 0 = derive it from rate and speed')
    parser.add_argument('--brush_radius_mm', type=float, default=1.5, help='half-width of the painted band')
    parser.add_argument('--noise_mm', type=float, default=0.3, help='tracker noise, per axis')
    parser.add_argument('--max_points_per_region', type=int, default=0, help='0 = keep every point')
    parser.add_argument('--wander', type=float, default=0.35,
                        help='0 = straight strokes, 1 = aimless; how much the direction drifts per step')
    parser.add_argument('--out_dir', default=osp.join(REPO_DIR, 'output', 'painted'))
    parser.add_argument('--preview', action='store_true', help='render a PNG of the painted shape')
    parser.add_argument('--no_save', action='store_true')
    return parser


def region_coverage(simulator, vertices, regions_points, radius_mm):
    r"""Fraction of each region's area that ended up within a brush radius of paint."""
    from scipy.spatial import cKDTree

    fractions = []
    for index, points in enumerate(regions_points):
        region = trimesh.Trimesh(vertices, simulator.region_faces[index], process=False)
        if len(points) == 0:
            fractions.append(0.0)
            continue
        painted = cKDTree(points).query(region.triangles_center)[0] <= radius_mm
        fractions.append(float(region.area_faces[painted].sum() / region.area))
    return fractions


def sample_spacing(args):
    r"""Distance between consecutive probe samples: speed / rate."""
    if args.spacing_mm:
        return args.spacing_mm
    return max(args.probe_speed_mm_s / max(args.probe_rate_hz, 1e-6), 1e-3)


class PaintSimulator:
    r"""Painted point clouds for the six landmark regions of an SSM shape."""

    def __init__(self, model_file=DEFAULT_MODEL, num_modes='95%'):
        self.shapes = LandmarkShapes(model_file, num_modes, points_per_region=1)
        self.faces = self.shapes.faces
        self.region_faces = self.shapes.region_faces
        self.region_vertices = [np.unique(f.reshape(-1)) for f in self.region_faces]
        self.neighbors = self._adjacency(self.faces, len(self.shapes.mean) // 3)

    @staticmethod
    def _adjacency(faces, num_vertices):
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.vstack([edges, edges[:, ::-1]])
        order = np.argsort(edges[:, 0], kind='stable')
        edges = edges[order]
        starts = np.searchsorted(edges[:, 0], np.arange(num_vertices))
        ends = np.searchsorted(edges[:, 0], np.arange(num_vertices), side='right')
        return [np.unique(edges[s:e, 1]) for s, e in zip(starts, ends)]

    def shape(self, seed):
        vertices, _ = self.shapes.shape(seed)          # millimetres
        return vertices, trimesh.Trimesh(vertices, self.faces, process=False)

    def _walk(self, vertices, allowed, start, length_mm, rng, wander):
        r"""Vertex path across the surface, biased to keep going the same way."""
        path = [start]
        current = start
        candidates = self.neighbors[current][np.isin(self.neighbors[current], allowed)]
        if len(candidates) == 0:
            return np.array(path)
        direction = vertices[rng.choice(candidates)] - vertices[current]
        direction /= max(np.linalg.norm(direction), 1e-9)
        travelled = 0.0
        while travelled < length_mm:
            options = self.neighbors[current][np.isin(self.neighbors[current], allowed)]
            options = options[options != path[-2]] if len(path) > 1 and len(options) > 1 else options
            if len(options) == 0:
                break
            steps = vertices[options] - vertices[current]
            norms = np.linalg.norm(steps, axis=1, keepdims=True)
            scores = (steps / np.maximum(norms, 1e-9)) @ direction
            # softmax pick: mostly forward, occasionally a turn, so strokes curve
            weights = np.exp((scores - scores.max()) / max(wander, 1e-3))
            nxt = int(rng.choice(options, p=weights / weights.sum()))
            step = vertices[nxt] - vertices[current]
            travelled += float(np.linalg.norm(step))
            direction = (1 - wander) * direction + wander * step / max(np.linalg.norm(step), 1e-9)
            direction /= max(np.linalg.norm(direction), 1e-9)
            path.append(nxt)
            current = nxt
        return np.array(path)

    @staticmethod
    def _smooth(points, window):
        r"""Moving average along a path, with the ends held."""
        window = int(window)
        if window <= 1 or len(points) <= 2:
            return points
        window = min(window, len(points))
        pad = window // 2
        padded = np.vstack([np.repeat(points[:1], pad, axis=0), points,
                            np.repeat(points[-1:], pad, axis=0)])
        kernel = np.ones(window) / window
        smoothed = np.stack([np.convolve(padded[:, i], kernel, mode='valid') for i in range(3)], axis=1)
        return smoothed[:len(points)]

    @staticmethod
    def _wobble(count, amplitude_mm, spacing_mm, rng, length_mm=6.0):
        r"""Slow lateral drift of the probe, in millimetres."""
        raw = rng.normal(size=(count, 3))
        # convolve('same') returns max(len(signal), len(kernel)), so never let the
        # smoothing window outrun the pass itself
        window = max(1, min(count, int(round(length_mm / max(spacing_mm, 1e-3)))))
        if window > 1:
            kernel = np.ones(window) / window
            raw = np.stack([np.convolve(raw[:, i], kernel, mode='same') for i in range(3)], axis=1)
            raw *= np.sqrt(window)                                  # convolution shrinks the variance
        return np.clip(raw * amplitude_mm, -3 * amplitude_mm, 3 * amplitude_mm)

    @staticmethod
    def _resample(points, spacing):
        r"""Even arc-length spacing along a polyline."""
        if len(points) < 2:
            return points
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(segments)])
        if arc[-1] < spacing:
            return points[:1]
        targets = np.arange(0.0, arc[-1], spacing)
        return np.stack([np.interp(targets, arc, points[:, i]) for i in range(3)], axis=1)

    def _sweep_lines(self, vertices, index, args):
        r"""Back-and-forth passes covering the region.

        The region is flattened onto its own principal plane, its faces are binned
        into bands one pass apart, and each band is walked end to end in
        alternating directions -- a lawnmower pattern over the surface itself, so
        every point stays on the mesh and inside the region.
        """
        region = trimesh.Trimesh(vertices, self.region_faces[index], process=False)
        centers = region.triangles_center
        origin = centers.mean(axis=0)
        axes = np.linalg.svd(centers - origin, full_matrices=False)[2]
        along = (centers - origin) @ axes[0]
        across = (centers - origin) @ axes[1]

        band_width = max(args.brush_radius_mm * args.sweep_overlap, 0.3)
        bands = np.floor((across - across.min()) / band_width).astype(int)
        lines = []
        for band in range(bands.max() + 1):
            selected = np.where(bands == band)[0]
            if len(selected) < 2:
                continue
            order = selected[np.argsort(along[selected])]
            if band % 2:                       # alternate, so passes join end to end
                order = order[::-1]
            lines.append(centers[order])
        return lines, region

    def paint_region_strokes(self, vertices, index, args, rng):
        r"""Ordered strokes for one region: each is the probe path, point by point."""
        allowed = self.region_vertices[index]
        region_mesh = trimesh.Trimesh(vertices, self.region_faces[index], process=False)
        extent = float(np.linalg.norm(region_mesh.extents))

        spacing = sample_spacing(args)
        if args.pattern == 'uniform':
            count = max(4, int(region_mesh.area / max(spacing ** 2, 1e-6)))
            points, _ = trimesh.sample.sample_surface(region_mesh, count, seed=int(rng.integers(1 << 31)))
            strokes = [np.asarray(points)]
        elif args.pattern == 'sweep':
            lines, region_mesh = self._sweep_lines(vertices, index, args)
            if args.coverage < 1 and len(lines) > 1:
                keep = max(1, int(round(len(lines) * args.coverage)))
                lines = [lines[i] for i in sorted(rng.choice(len(lines), keep, replace=False))]
            strokes = []
            for line in lines:
                # face centres zigzag within a band; smooth before sampling so the
                # pass reads as one sweep rather than a saw
                step = float(np.median(np.linalg.norm(np.diff(line, axis=0), axis=1))) if len(line) > 1 else 1.0
                line = self._smooth(line, max(1, round(args.pass_smoothing / max(step, 1e-3))))
                resampled = self._resample(line, spacing)
                if len(resampled) < 2:
                    continue
                if args.coverage < 1:      # a pass may also stop short of the far edge
                    span = rng.uniform(args.coverage, 1.0)
                    length = max(2, int(len(resampled) * span))
                    start = rng.integers(0, len(resampled) - length + 1)
                    resampled = resampled[start:start + length]
                resampled = resampled + self._wobble(len(resampled), args.wobble_mm, spacing, rng,
                                                     args.wobble_length_mm)
                band, _, _ = trimesh.proximity.closest_point(region_mesh, resampled)
                strokes.append(np.asarray(band))
        else:
            strokes = []
            for _ in range(args.strokes_per_region):
                start = int(rng.choice(allowed))
                path = self._walk(vertices, allowed, start, min(args.stroke_length_mm, extent), rng, args.wander)
                if len(path) < 2:
                    continue
                line = self._resample(vertices[path], spacing)
                # the probe wobbles smoothly across the brush width, so correlate the
                # offsets along the stroke instead of drawing them independently
                offsets = self._wobble(len(line), args.brush_radius_mm / 2, spacing, rng,
                                       args.wobble_length_mm)
                band = offsets + line
                band, _, _ = trimesh.proximity.closest_point(region_mesh, band)
                strokes.append(np.asarray(band))

        out = []
        for stroke in strokes:
            if args.noise_mm > 0:
                stroke = stroke + rng.normal(scale=args.noise_mm, size=stroke.shape)
            if args.max_points_per_region:
                keep = max(2, args.max_points_per_region // max(len(strokes), 1))
                if len(stroke) > keep:      # thin evenly so the path order survives
                    stroke = stroke[np.linspace(0, len(stroke) - 1, keep).astype(int)]
            out.append(stroke)
        return out

    def paint_region(self, mesh, vertices, index, args, rng):
        strokes = self.paint_region_strokes(vertices, index, args, rng)
        return np.vstack(strokes) if strokes else np.empty((0, 3))

    def paint_detailed(self, seed, args):
        r"""Strokes per region, keeping the order the probe visited them in."""
        vertices, _ = self.shape(seed)
        rng = np.random.default_rng(seed + 31_000_000)
        return [self.paint_region_strokes(vertices, i, args, rng)
                for i in range(len(self.region_faces))], vertices

    def paint(self, seed, args):
        r"""One painted cloud per region, in millimetres."""
        vertices, mesh = self.shape(seed)
        rng = np.random.default_rng(seed + 31_000_000)
        return [self.paint_region(mesh, vertices, i, args, rng) for i in range(len(self.region_faces))], vertices


def render_preview(simulator, vertices, regions, path):
    import open3d as o3d

    shell = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                      o3d.utility.Vector3iVector(simulator.faces))
    shell = shell.simplify_quadric_decimation(1200)
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(shell)
    wire.paint_uniform_color([0.72, 0.75, 0.78])

    cloud = o3d.geometry.PointCloud()
    points = np.vstack([r for r in regions if len(r)])
    colors = np.vstack([np.tile(REGION_COLORS[i % len(REGION_COLORS)], (len(r), 1))
                        for i, r in enumerate(regions) if len(r)])
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1400, height=900, visible=False)
    vis.add_geometry(wire)
    vis.add_geometry(cloud)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.988, 0.988, 0.984])
    opt.point_size = 5.0
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()
    return path


def main():
    args = make_parser().parse_args()
    simulator = PaintSimulator(args.model_file)
    os.makedirs(args.out_dir, exist_ok=True)

    for offset in range(args.num_shapes):
        seed = args.seed + offset
        regions, vertices = simulator.paint(seed, args)
        counts = [len(r) for r in regions]
        mesh = trimesh.Trimesh(vertices, simulator.faces, process=False)
        distance = np.abs(trimesh.proximity.signed_distance(
            mesh, np.vstack([r for r in regions if len(r)])))
        coverage = region_coverage(simulator, vertices, regions, args.brush_radius_mm)
        print('seed {}: {} points {}  ({:.0f} Hz at {:.0f} mm/s -> {:.2f} mm spacing, '
              'median {:.2f} mm off the surface)'.format(
                  seed, sum(counts), counts, args.probe_rate_hz, args.probe_speed_mm_s,
                  sample_spacing(args), float(np.median(distance))), flush=True)
        print('  region coverage: {}  (mean {:.0f}%)'.format(
            ' '.join('{:.0f}%'.format(100 * c) for c in coverage),
            100 * float(np.mean(coverage))), flush=True)

        if not args.no_save:
            np.savez_compressed(
                osp.join(args.out_dir, 'painted_{}.npz'.format(seed)),
                points=np.vstack([r for r in regions if len(r)]),
                region_index=np.concatenate([np.full(len(r), i) for i, r in enumerate(regions) if len(r)]),
                vertices=vertices.astype(np.float32), faces=simulator.faces,
                settings=np.array(str(vars(args))))
        if args.preview:
            path = render_preview(simulator, vertices, regions,
                                  osp.join(args.out_dir, 'painted_{}.png'.format(seed)))
            print('  preview -> ' + path, flush=True)

    if not args.no_save:
        print('wrote to ' + args.out_dir)


if __name__ == '__main__':
    main()
