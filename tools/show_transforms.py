r"""Apply the saved transforms to the raw CSVs and look at the result (Open3D window).

Reads `transforms.json` written by `case_viewer.py`, loads each scenario's CSV in the
tracker frame, applies its `tracker_to_ct` matrix, and draws the points on the bone.
Nothing is recomputed here: this is what the stored poses actually do.

Left station is before: the raw CSV as digitized, carrying its own tracker orientation,
shifted onto the bone so it is visible at all (the raw coordinates sit ~600 mm away).
Right station is after: the same points under the stored transform.

    python tools/show_transforms.py
    python tools/show_transforms.py --transforms output/cases/<case>/transforms.json

    n / p   next / previous scenario     a   all scenarios at once
    d       colour by distance to the surface  <-> one colour per scenario
    g       ghost wireframe on / off     h   this help
"""

import argparse
import json
import os.path as osp
import sys

import numpy as np
import open3d as o3d
import trimesh

REPO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, osp.join(REPO_DIR, 'tools'))

from selfpair_viewer import sphere_cloud, ghost_wireframe, text_mesh, GHOST_COLOR, TEXT_COLOR  # noqa: E402
from simulate_painting import REGION_COLORS  # noqa: E402

DEFAULT = osp.join(REPO_DIR, 'output', 'cases', 'S260655_LEFT', 'transforms.json')
RAMP = np.array([[0.804, 0.886, 0.984], [0.431, 0.655, 0.925],
                 [0.165, 0.471, 0.839], [0.063, 0.259, 0.506]])

HELP = """
  n / p  next / previous scenario      a  all scenarios at once
  d      distance colouring on / off   g  ghost wireframe on / off
  h      this help
"""


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transforms', default=DEFAULT, help='transforms.json from case_viewer.py')
    parser.add_argument('--mesh', default=None, help='override the bone recorded in the file')
    parser.add_argument('--case_dir', default=None, help='override the CSV folder recorded in the file')
    parser.add_argument('--marker_mm', type=float, default=0.9, help='radius of the drawn points')
    parser.add_argument('--vmax_mm', type=float, default=3.0, help='top of the distance colour ramp')
    parser.add_argument('--ghost_faces', type=int, default=1600)
    parser.add_argument('--selftest', action='store_true', help='load and report, no window')
    return parser


def ramp_colors(values, vmax):
    t = np.clip(values / max(vmax, 1e-9), 0, 0.999) * (len(RAMP) - 1)
    low = np.floor(t).astype(int)
    frac = (t - low)[:, None]
    return RAMP[low] * (1 - frac) + RAMP[low + 1] * frac


def main():
    args = make_parser().parse_args()
    with open(args.transforms) as f:
        document = json.load(f)
    case_dir = args.case_dir or document['case_dir']
    mesh = trimesh.load(args.mesh or document['mesh'], process=False)
    mesh.merge_vertices()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)))))

    cases = []
    for entry in document['scenarios']:
        raw = np.loadtxt(osp.join(case_dir, entry['file']), delimiter=',', ndmin=2)[:, :3]
        matrix = np.asarray(entry['tracker_to_ct'], dtype=np.float64)
        moved = raw @ matrix[:3, :3].T + matrix[:3, 3]
        distance = scene.compute_distance(o3d.core.Tensor(moved.astype(np.float32))).numpy()
        # the raw cloud lives hundreds of mm off; park it on the bone so both
        # stations frame the same way, orientation untouched
        before = raw - raw.mean(axis=0) + np.asarray(mesh.vertices).mean(axis=0)
        cases.append({'name': entry['scenario'], 'points': moved, 'before': before,
                      'distance': distance, 'rms': float(np.sqrt(np.mean(distance ** 2)))})
        print('{:<8} {:>3} points   surface RMS {:6.2f} mm   max {:6.2f} mm'.format(
            entry['scenario'], len(moved), cases[-1]['rms'], distance.max()), flush=True)
    print('\n{} ({} scenarios, {})'.format(osp.basename(args.transforms), len(cases),
                                           document.get('method', 'pose')), flush=True)
    if args.selftest:
        return

    state = {'index': 0, 'all': False, 'distance': True, 'ghost': True}
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    span = np.ptp(vertices, axis=0).max()
    offset = np.array([1.5 * span, 0.0, 0.0])          # before on the left, after on the right
    points_mesh, points_after = o3d.geometry.TriangleMesh(), o3d.geometry.TriangleMesh()
    shell, shell_after = o3d.geometry.LineSet(), o3d.geometry.LineSet()

    def bone(shift):
        surface = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices + shift),
                                            o3d.utility.Vector3iVector(faces))
        surface.compute_vertex_normals()
        surface.paint_uniform_color([0.88, 0.88, 0.86])
        return surface

    surface, surface_after = bone(np.zeros(3)), bone(offset)
    labels = []

    def shown():
        return cases if state['all'] else [cases[state['index']]]

    def refresh(vis=None):
        chosen = shown()
        points = np.vstack([c['points'] for c in chosen])
        if state['distance']:
            colors = ramp_colors(np.concatenate([c['distance'] for c in chosen]), args.vmax_mm)
        else:
            colors = np.vstack([np.tile(REGION_COLORS[i % len(REGION_COLORS)], (len(c['points']), 1))
                                for i, c in enumerate(chosen)])
        before = np.vstack([c['before'] for c in chosen])
        for geom, pts, cols in ((points_mesh, before, np.tile([0.922, 0.408, 0.204], (len(before), 1))),
                                (points_after, points + offset, colors)):
            blob = sphere_cloud(pts, cols, args.marker_mm)
            geom.vertices, geom.triangles = blob.vertices, blob.triangles
            geom.vertex_colors, geom.vertex_normals = blob.vertex_colors, blob.vertex_normals

        for ls, shift in ((shell, np.zeros(3)), (shell_after, offset)):
            if state['ghost']:
                wire = ghost_wireframe(vertices + shift, faces, args.ghost_faces, GHOST_COLOR)
                ls.points, ls.lines, ls.colors = wire.points, wire.lines, wire.colors
            else:
                ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                ls.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))

        anchor = vertices.min(axis=0) + np.array([0.0, 1.15 * span, 0.0])
        if state['all']:
            lines = ['all {} scenarios'.format(len(cases)),
                     'surface RMS {:.2f} .. {:.2f} mm'.format(min(c['rms'] for c in cases),
                                                              max(c['rms'] for c in cases))]
        else:
            case = chosen[0]
            lines = ['{}  [{}/{}]'.format(case['name'], state['index'] + 1, len(cases)),
                     'surface RMS {:.2f} mm   max {:.2f} mm'.format(case['rms'], case['distance'].max())]
        new = [text_mesh(['BEFORE: as digitized'] + lines, 0.05 * span, TEXT_COLOR, anchor),
               text_mesh(['AFTER: stored transform'], 0.05 * span, TEXT_COLOR, anchor + offset)]
        if vis is not None:
            for old in labels:
                vis.remove_geometry(old, reset_bounding_box=False)
            for label in new:
                vis.add_geometry(label, reset_bounding_box=False)
            for geom in (points_mesh, points_after, shell, shell_after):
                vis.update_geometry(geom)
        labels[:] = new
        if not state['all']:
            print('[{}/{}] {}  surface RMS {:.2f} mm'.format(
                state['index'] + 1, len(cases), chosen[0]['name'], chosen[0]['rms']), flush=True)

    refresh()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='saved transforms on the bone', width=1440, height=900)
    for geom in (surface, surface_after, points_mesh, points_after, shell, shell_after):
        vis.add_geometry(geom)
    for label in labels:
        vis.add_geometry(label)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.988, 0.988, 0.984])
    opt.mesh_show_back_face = True

    def step(delta, vis):
        state['all'] = False
        state['index'] = (state['index'] + delta) % len(cases)
        refresh(vis)
        return True

    def toggle(key, vis):
        state[key] = not state[key]
        refresh(vis)
        return True

    vis.register_key_callback(ord('N'), lambda v: step(1, v))
    vis.register_key_callback(ord('P'), lambda v: step(-1, v))
    vis.register_key_callback(ord('A'), lambda v: toggle('all', v))
    vis.register_key_callback(ord('D'), lambda v: toggle('distance', v))
    vis.register_key_callback(ord('G'), lambda v: toggle('ghost', v))
    vis.register_key_callback(ord('H'), lambda v: (print(HELP), True)[1])
    print(HELP)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
