"""Generate a left knee URDF and real STL meshes by reflecting right CAD local X.

The mounting preview's left rotation maps this reflection to the anatomical
sagittal mirror. This is distinct from reversing only a joint axis.
"""

import argparse
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
REFLECTION = np.diag([-1.0, 1.0, 1.0])
STL_TRIANGLE = np.dtype([
    ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
])


def vector(text):
    values = np.fromstring(text, sep=" ")
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"Expected three finite values: {text}")
    return values


def numbers(values):
    return " ".join(f"{0.0 if abs(v) < 1e-15 else v:.12g}" for v in values)


def mirror_stl(source, target):
    """Reflect vertices and normals, reverse winding; support binary and ASCII STL."""
    raw = Path(source).read_bytes()
    if len(raw) >= 84 and 84 + 50 * struct.unpack_from("<I", raw, 80)[0] == len(raw):
        triangles = np.frombuffer(raw, dtype=STL_TRIANGLE, offset=84).copy()
    else:
        vertices = []
        for line in raw.decode("ascii").splitlines():
            tokens = line.split()
            if tokens and tokens[0].lower() == "vertex":
                vertices.append([float(x) for x in tokens[1:]])
        if not vertices or len(vertices) % 3:
            raise ValueError(f"Invalid STL: {source}")
        triangles = np.zeros(len(vertices) // 3, dtype=STL_TRIANGLE)
        triangles["vertices"] = np.asarray(vertices).reshape(-1, 3, 3)
    triangles["vertices"] = triangles["vertices"] @ REFLECTION
    triangles["vertices"][:, [1, 2]] = triangles["vertices"][:, [2, 1]]
    # Recompute normals after reflection + winding reversal (some CAD normals
    # are zero or stale). Degenerate triangles retain zero normals.
    v = triangles["vertices"].astype(float)
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    np.divide(normals, lengths[:, None], out=normals, where=lengths[:, None] > 0)
    triangles["normal"] = normals
    header = b"Left knee mesh: right CAD local X reflection; reversed winding".ljust(80, b" ")
    Path(target).write_bytes(header + struct.pack("<I", len(triangles)) + triangles.tobytes())


def mirror(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Source and target must differ")
    package = source.parent.parent
    root = ET.parse(source).getroot()
    for origin in root.findall(".//origin"):
        origin.set("xyz", numbers(REFLECTION @ vector(origin.get("xyz", "0 0 0"))))
        r = Rotation.from_euler("xyz", vector(origin.get("rpy", "0 0 0"))).as_matrix()
        origin.set("rpy", numbers(Rotation.from_matrix(REFLECTION @ r @ REFLECTION).as_euler("xyz")))
    for inertia in root.findall(".//inertia"):
        matrix = np.array([
            [float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
            [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
            [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))],
        ])
        matrix = REFLECTION @ matrix @ REFLECTION
        for key, i, j in (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
                          ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2)):
            inertia.set(key, f"{matrix[i, j]:.12g}")
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        axis = joint.find("axis")
        if axis is None:
            axis = ET.SubElement(joint, "axis", xyz="1 0 0")
        value = REFLECTION @ vector(axis.get("xyz", "1 0 0"))
        if joint.get("type") in ("revolute", "continuous"):
            # Rotation axes are axial vectors: det(S) * S * axis. Keeping
            # the angle limits therefore gives mirrored motion at equal q.
            value = -value
        elif joint.get("type") != "prismatic":
            raise ValueError(f"Unsupported joint type: {joint.get('type')}")
        axis.set("xyz", numbers(value))
    meshes = {}
    prefix = f"package://{package.name}/"
    for mesh in root.findall(".//mesh"):
        uri = mesh.get("filename")
        if not uri.startswith(prefix):
            raise ValueError(f"Expected local package mesh URI: {uri}")
        original = package / uri[len(prefix):]
        mirrored = original.with_name(original.stem + "_L" + original.suffix)
        if original not in meshes:
            mirror_stl(original, mirrored)
            meshes[original] = mirrored
        mesh.set("filename", prefix + mirrored.relative_to(package).as_posix())
    root.insert(0, ET.Comment(
        f" Generated from {source.name} by tools/mirror_knee_urdf.py: CAD local X reflection. "
        "Meshes, COM, inertia, transforms and joint axes mirrored; masses/limits preserved. "
    ))
    ET.indent(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    print(f"Generated {target} and {len(meshes)} mirrored STL files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "urdf/knee_exo_hinge/urdf/knee_exo_hinge_r.urdf")
    parser.add_argument("--target", type=Path, default=ROOT / "urdf/knee_exo_hinge/urdf/knee_exo_hinge_l.urdf")
    args = parser.parse_args()
    mirror(args.source, args.target)


if __name__ == "__main__":
    main()
