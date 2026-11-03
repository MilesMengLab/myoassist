"""Check actual mesh-surface intersections, independent of MuJoCo collision filtering.

This checks sampled kinematic poses against rendered bone meshes, not skin,
muscle/tendon paths, strap fit, or convex-hull contact fidelity.
"""

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

from tools.preview_knee_mount import DEFAULT_CONFIG, DEFAULT_OUTPUT, build, load_config, update


def mesh_geometry(model, data, geom):
    mesh = int(model.geom_dataid[geom])
    start = int(model.mesh_vertadr[mesh])
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
    vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    start = int(model.mesh_faceadr[mesh])
    faces = model.mesh_face[start:start + model.mesh_facenum[mesh]]
    return vertices, faces


def edge_hits(vertices, faces, triangles):
    centers = triangles.mean(axis=1)
    radius = float(np.linalg.norm(triangles - centers[:, None], axis=2).max())
    tree = cKDTree(centers)
    edges = np.unique(np.sort(np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
    ]), axis=1), axis=0)
    lower, upper = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
    points = vertices[edges]
    edges = edges[np.all(points.max(axis=1) >= lower, axis=1) & np.all(points.min(axis=1) <= upper, axis=1)]
    hits = 0
    for a, b in edges:
        origin = vertices[a]
        direction = vertices[b] - origin
        candidates = tree.query_ball_point((origin + vertices[b]) / 2, radius + np.linalg.norm(direction) / 2)
        if not candidates:
            continue
        tri = triangles[candidates]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        h = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, h)
        mask = np.abs(det) > 1e-12
        if not np.any(mask):
            continue
        e1, e2, h, inv = e1[mask], e2[mask], h[mask], 1 / det[mask]
        delta = origin - tri[mask, 0]
        u = inv * np.einsum("ij,ij->i", delta, h)
        q = np.cross(delta, e1)
        v = inv * (q @ direction)
        t = inv * np.einsum("ij,ij->i", e2, q)
        hits += bool(np.any((u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)))
    return hits


def surface_crossings(first, second):
    va, fa = first
    vb, fb = second
    if np.any(va.max(axis=0) < vb.min(axis=0)) or np.any(vb.max(axis=0) < va.min(axis=0)):
        return 0
    return edge_hits(va, fa, vb[fb]) + edge_hits(vb, fb, va[fa])


def _point_segment_sq(point, start, end):
    edge = end - start
    norm = np.sum(edge * edge, axis=-1)
    t = np.divide(np.sum((point - start) * edge, axis=-1), norm,
                  out=np.zeros_like(norm), where=norm > 1e-24)
    delta = point - (start + np.clip(t, 0, 1)[..., None] * edge)
    return np.sum(delta * delta, axis=-1)


def _point_triangle_sq(point, tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    norm = np.sum(normal * normal, axis=1)
    signed = np.sum((point - a) * normal, axis=1)
    factor = np.divide(signed, norm, out=np.zeros_like(norm), where=norm > 1e-24)
    projected = point - factor[:, None] * normal
    inside = norm > 1e-24
    for start, end in ((a, b), (b, c), (c, a)):
        inside &= np.sum(np.cross(end - start, projected - start) * normal, axis=1) >= -1e-20
    edge_distance = np.minimum.reduce([_point_segment_sq(point, a, b),
                                       _point_segment_sq(point, b, c),
                                       _point_segment_sq(point, c, a)])
    return np.where(inside, factor * signed, edge_distance)


def surface_distance(first, second):
    """Minimum triangle-surface distance for disjoint meshes, in meters.

    Call surface_crossings separately before interpreting this as clearance.
    Includes vertex/face and interior edge/edge minima, not just nearest vertices.
    """
    va, fa = first
    vb, fb = second
    upper = float(cKDTree(vb).query(va)[0].min())
    triangles = vb[fb]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    tree = cKDTree(centers)
    max_radius = float(radii.max())
    best_sq = upper * upper
    for tri in va[fa]:
        center = tri.mean(axis=0)
        radius = float(np.linalg.norm(tri - center, axis=1).max())
        ids = tree.query_ball_point(center, radius + max_radius + np.sqrt(best_sq))
        if not ids:
            continue
        candidates = triangles[ids]
        dist = []
        for i in range(3):
            dist.append(_point_triangle_sq(tri[i], candidates))
            dist.append(_point_triangle_sq(candidates[:, i], np.broadcast_to(tri, candidates.shape)))
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            u = b - a
            for j in range(3):
                c, d = candidates[:, j], candidates[:, (j + 1) % 3]
                v, w = d - c, a - c
                aa = np.dot(u, u)
                bb, cc = v @ u, np.sum(v * v, axis=1)
                dd, ee = w @ u, np.sum(v * w, axis=1)
                denominator = aa * cc - bb * bb
                s = np.divide(bb * ee - cc * dd, denominator,
                              out=np.zeros_like(cc), where=denominator > 1e-24)
                t = np.divide(aa * ee - bb * dd, denominator,
                              out=np.zeros_like(cc), where=denominator > 1e-24)
                delta = w + s[:, None] * u - t[:, None] * v
                valid = (denominator > 1e-24) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
                dist.append(np.where(valid, np.sum(delta * delta, axis=1), np.inf))
        best_sq = min(best_sq, float(np.min(dist)))
    return float(np.sqrt(max(0, best_sq)))


def audit(config_path=DEFAULT_CONFIG, angles=(0, 30, 60, 90)):
    model, data, devices = build()
    config = load_config(config_path)
    rows = []
    # Every human mesh is checked, including the opposite leg and pelvis.
    device_geoms = [model.geom(f"knee_{s}_{p}_visual").id for s in ("R", "L") for p in ("upper", "lower")]
    human_geoms = [g for g in range(model.ngeom) if model.geom_type[g] == 7 and g not in device_geoms]
    for angle in angles:
        alignment = update(model, data, devices, config, angle)
        meshes = {g: mesh_geometry(model, data, g) for g in human_geoms + device_geoms}
        crossings = []
        for device in device_geoms:
            for human in human_geoms:
                count = surface_crossings(meshes[device], meshes[human])
                if count:
                    crossings.append({"device": model.geom(device).name,
                                      "human": model.geom(human).name, "crossing_edges": int(count)})
        row = {"flexion_deg": angle, "surface_intersections": crossings, "axis_alignment": alignment}
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {"mounting": config, "poses": rows,
              "scope": "actual mesh surfaces only; no skin or line-muscle clearance certification"}
    path = DEFAULT_OUTPUT / "clearance_report.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    audit(args.config)


if __name__ == "__main__":
    main()
