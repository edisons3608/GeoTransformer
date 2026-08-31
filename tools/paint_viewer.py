r"""Live viewer for simulated painting (Open3D window, mouse-orbit).

Shows the talus as a faint wireframe with the painted points inside each of the
six landmark regions, joined in the order the probe visited them so each stroke
reads as a path rather than a cloud.

    python tools/paint_viewer.py
    python tools/paint_viewer.py --seed 909 --strokes_per_region 3 --wander 0.15

Keys:
    space   replay the painting, point by point       a  show everything at once
    n / p   next / previous shape                     r  repaint this shape
    l       stroke lines on / off                     g  ghost wireframe on / off
    u       paint-region underlay on / off            h  this help
"""

import argparse
import os.path as osp
import sys
import time

import numpy as np
import open3d as o3d

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

from simulate_painting import PaintSimulator, REGION_COLORS, make_parser as paint_parser  # noqa: E402

GHOST_COLOR = np.array([0.72, 0.75, 0.78])

HELP = """
  space  replay the painting stroke by stroke    a  show all points
  n / p  next / previous shape                   r  repaint this shape
  l      stroke lines on / off                   g  ghost wireframe on / off
  u      paint-region underlay on / off          h  this help
"""


def make_viewer_parser():
    parser = paint_parser()
    parser.add_argument('--ghost_faces', type=int, default=1200)
    parser.add_argument('--replay_rate', type=float, default=25.0, help='points revealed per second')
    parser.add_argument('--underlay_tint', type=float, default=0.72,
                        help='0 = solid region colour, 1 = invisible; stands in for translucency')
    parser.add_argument('--selftest', action='store_true')
    return parser


def main():
    args = make_viewer_parser().parse_args()
    simulator = PaintSimulator(args.model_file)

    state = {'seed': args.seed, 'repaint': 0, 'revealed': None, 'playing': False,
             'lines': True, 'ghost': True, 'underlay': True, 'last': 0.0}
    cache = {}

    def painting():
        r"""Strokes for the current shape, flattened with their stroke and region ids."""
        key = (state['seed'], state['repaint'])
        if key not in cache:
            regions, _ = simulator.paint_detailed(state['seed'] + state['repaint'] * 7919, args)
            points, colors, segments, offset = [], [], [], 0
            for region_index, strokes in enumerate(regions):
                color = REGION_COLORS[region_index % len(REGION_COLORS)]
                for stroke in strokes:
                    points.append(stroke)
                    colors.append(np.tile(color, (len(stroke), 1)))
                    segments.append(np.stack([np.arange(offset, offset + len(stroke) - 1),
                                              np.arange(offset + 1, offset + len(stroke))], axis=1))
                    offset += len(stroke)
            # a repaint redraws on the same shape, so take vertices from the base seed
            vertices, _ = simulator.shape(state['seed'])
            cache[key] = {'points': np.vstack(points), 'colors': np.vstack(colors),
                          'segments': np.vstack(segments) if segments else np.zeros((0, 2), int),
                          'vertices': vertices,
                          'counts': [sum(len(s) for s in strokes) for strokes in regions],
                          'strokes': sum(len(s) for s in regions)}
        return cache[key]

    cloud = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()
    ghost = o3d.geometry.LineSet()
    underlay = o3d.geometry.TriangleMesh()

    def build_underlay(vertices):
        """The six paint regions as pale surface patches under the points.

        The legacy renderer has no alpha, so the region colour is blended toward
        the background instead -- it reads as a translucent wash and keeps the
        painted points on top clearly visible.
        """
        tris, colors = [], []
        for index, faces in enumerate(simulator.region_faces):
            tint = REGION_COLORS[index % len(REGION_COLORS)]
            tint = tint * (1 - args.underlay_tint) + args.underlay_tint
            for face in faces:
                tris.append(face)
                colors.append(tint)
        flat = vertices[np.asarray(tris)].reshape(-1, 3)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(flat),
            o3d.utility.Vector3iVector(np.arange(len(flat)).reshape(-1, 3)))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(np.asarray(colors), 3, axis=0))
        mesh.compute_vertex_normals()
        return mesh

    def refresh_underlay(vis=None):
        data = painting()
        if state['underlay']:
            built = build_underlay(data['vertices'])
            underlay.vertices = built.vertices
            underlay.triangles = built.triangles
            underlay.vertex_colors = built.vertex_colors
            underlay.vertex_normals = built.vertex_normals
        else:
            underlay.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            underlay.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
        if vis is not None:
            vis.update_geometry(underlay)

    def refresh(vis=None):
        data = painting()
        shown = data['points'].shape[0] if state['revealed'] is None else int(state['revealed'])
        cloud.points = o3d.utility.Vector3dVector(data['points'][:shown])
        cloud.colors = o3d.utility.Vector3dVector(data['colors'][:shown])

        keep = (data['segments'][(data['segments'] < shown).all(axis=1)]
                if state['lines'] and shown > 1 else np.zeros((0, 2), dtype=np.int32))
        if len(keep) == 0:
            keep = np.zeros((1, 2), dtype=np.int32)   # zero-length: an empty LineSet warns every frame
        lines.points = o3d.utility.Vector3dVector(data['points'])
        lines.lines = o3d.utility.Vector2iVector(keep)
        lines.colors = o3d.utility.Vector3dVector(data['colors'][keep[:, 0]] * 0.85)

        if vis is not None:
            vis.update_geometry(cloud)
            vis.update_geometry(lines)

    def rebuild_ghost(vis=None):
        data = painting()
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']),
                                         o3d.utility.Vector3iVector(simulator.faces))
        if len(simulator.faces) > args.ghost_faces:
            mesh = mesh.simplify_quadric_decimation(int(args.ghost_faces))
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wire.paint_uniform_color(GHOST_COLOR)
        if state['ghost']:
            ghost.points, ghost.lines = wire.points, wire.lines
            ghost.colors = wire.colors
        else:
            ghost.points = o3d.utility.Vector3dVector(np.asarray(wire.points)[:1])
            ghost.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        if vis is not None:
            vis.update_geometry(ghost)

    def announce():
        data = painting()
        print('\nseed {}{}  --  {} points in {} strokes  {}'.format(
            state['seed'], '' if not state['repaint'] else ' (repaint {})'.format(state['repaint']),
            len(data['points']), data['strokes'], data['counts']), flush=True)

    announce()
    refresh()
    rebuild_ghost()
    refresh_underlay()
    if args.selftest:
        for _ in range(3):
            state['seed'] += 1
            announce()
            refresh()
            rebuild_ghost()
        print('\nselftest ok', flush=True)
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Simulated painting', width=1400, height=900)
    vis.add_geometry(ghost)
    vis.add_geometry(underlay)
    vis.add_geometry(cloud)
    vis.add_geometry(lines)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.988, 0.988, 0.984])
    opt.point_size = 5.0
    opt.mesh_show_back_face = True

    def reload(vis):
        announce()
        state['revealed'] = None
        state['playing'] = False
        refresh(vis)
        rebuild_ghost(vis)
        refresh_underlay(vis)
        return True

    def step_shape(delta, vis):
        state['seed'] += delta
        state['repaint'] = 0
        return reload(vis)

    def repaint(vis):
        state['repaint'] += 1
        return reload(vis)

    def replay(vis):
        state['revealed'] = 0
        state['playing'] = True
        state['last'] = 0.0
        refresh(vis)
        return True

    def show_all(vis):
        state['revealed'] = None
        state['playing'] = False
        refresh(vis)
        return True

    def toggle_lines(vis):
        state['lines'] = not state['lines']
        refresh(vis)
        return True

    def toggle_ghost(vis):
        state['ghost'] = not state['ghost']
        rebuild_ghost(vis)
        return True

    def animate(vis):
        if not state['playing']:
            return False
        now = time.time()
        if state['last'] == 0.0:
            state['last'] = now
        total = len(painting()['points'])
        state['revealed'] = min(total, state['revealed'] + max(1, args.replay_rate * (now - state['last'])))
        state['last'] = now
        if state['revealed'] >= total:
            state['playing'] = False
            state['revealed'] = None
        refresh(vis)
        return True

    vis.register_key_callback(ord(' '), replay)
    vis.register_key_callback(ord('A'), show_all)
    vis.register_key_callback(ord('N'), lambda v: step_shape(1, v))
    vis.register_key_callback(ord('P'), lambda v: step_shape(-1, v))
    vis.register_key_callback(ord('R'), repaint)
    vis.register_key_callback(ord('L'), toggle_lines)
    vis.register_key_callback(ord('G'), toggle_ghost)
    def toggle_underlay(vis):
        state['underlay'] = not state['underlay']
        refresh_underlay(vis)
        return True

    vis.register_key_callback(ord('U'), toggle_underlay)
    vis.register_key_callback(ord('H'), lambda v: (print(HELP), True)[1])
    vis.register_animation_callback(animate)

    print(HELP)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
