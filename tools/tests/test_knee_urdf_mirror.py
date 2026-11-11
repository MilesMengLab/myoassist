"""Verify the generated left URDF is a geometric and inertial reflection."""

from pathlib import Path
import xml.etree.ElementTree as ET
from unittest import TestCase

import numpy as np

from tools.mirror_knee_urdf import REFLECTION, STL_TRIANGLE, vector


PACKAGE = Path(__file__).resolve().parents[2] / "urdf/knee_exo_hinge"


class KneeUrdfMirrorTest(TestCase):
    def test_stl_vertices_and_winding(self):
        for name in ("base_link", "motor_housing_link", "motor_output_link", "shank_link"):
            with self.subTest(mesh=name):
                right = np.frombuffer((PACKAGE / f"meshes/{name}.STL").read_bytes(), dtype=STL_TRIANGLE, offset=84)
                left = np.frombuffer((PACKAGE / f"meshes/{name}_L.STL").read_bytes(), dtype=STL_TRIANGLE, offset=84)
                expected = (right["vertices"] @ REFLECTION)[:, [0, 2, 1]]
                np.testing.assert_array_equal(left["vertices"], expected)
                vertices = left["vertices"].astype(float)
                normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
                self.assertTrue(np.all(np.einsum("ij,ij->i", normals, left["normal"]) >= -1e-12))

    def test_mass_com_and_inertia(self):
        right = ET.parse(PACKAGE / "urdf/knee_exo_hinge_r.urdf").getroot()
        left = ET.parse(PACKAGE / "urdf/knee_exo_hinge_l.urdf").getroot()
        for r, l in zip(right.findall("link"), left.findall("link")):
            with self.subTest(link=r.get("name")):
                self.assertEqual(r.find("inertial/mass").get("value"), l.find("inertial/mass").get("value"))
                np.testing.assert_allclose(
                    vector(l.find("inertial/origin").get("xyz")),
                    REFLECTION @ vector(r.find("inertial/origin").get("xyz")), atol=1e-12,
                )
                tensors = []
                for link in (r, l):
                    a = {k: float(v) for k, v in link.find("inertial/inertia").attrib.items()}
                    tensors.append(np.array([[a["ixx"], a["ixy"], a["ixz"]],
                                             [a["ixy"], a["iyy"], a["iyz"]],
                                             [a["ixz"], a["iyz"], a["izz"]]]))
                np.testing.assert_allclose(tensors[1], REFLECTION @ tensors[0] @ REFLECTION, atol=1e-12)
                self.assertTrue(np.all(np.linalg.eigvalsh(tensors[1]) > 0))
        for r, l in zip(right.findall("joint"), left.findall("joint")):
            np.testing.assert_allclose(vector(l.find("axis").get("xyz")),
                                       -REFLECTION @ vector(r.find("axis").get("xyz")), atol=1e-12)
            self.assertEqual(r.find("limit").attrib, l.find("limit").attrib)
        for mesh in left.findall(".//mesh"):
            self.assertIn("_L.STL", mesh.get("filename"))
