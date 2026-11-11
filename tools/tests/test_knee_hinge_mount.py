"""Articulated mounting must preserve motor/shank frames and mirror real motion."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import xml.etree.ElementTree as ET

import numpy as np

from tools.check_knee_mount_clearance import mesh_geometry, surface_crossings, surface_distance
from tools.mirror_knee_urdf import REFLECTION, STL_TRIANGLE, vector
from tools.preview_knee_hinge_mount import LINKS, PACKAGE, build, load_config, update


class KneeHingeMountTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.model, cls.data, cls.devices = build(Path(cls.temp.name))
        cls.config = load_config()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_mirrored_meshes_and_unique_joint_tree(self):
        right = ET.parse(PACKAGE / "urdf/knee_exo_hinge_r.urdf").getroot()
        left = ET.parse(PACKAGE / "urdf/knee_exo_hinge_l.urdf").getroot()
        for xml in (right, left):
            self.assertEqual(len(xml.findall("joint")), 3)
            self.assertEqual(len({j.get("name") for j in xml.findall("joint")}), 3)
            self.assertEqual(len({j.find("child").get("link") for j in xml.findall("joint")}), 3)
        for name in LINKS:
            r = np.frombuffer((PACKAGE / f"meshes/{name}.STL").read_bytes(), dtype=STL_TRIANGLE, offset=84)
            l = np.frombuffer((PACKAGE / f"meshes/{name}_L.STL").read_bytes(), dtype=STL_TRIANGLE, offset=84)
            np.testing.assert_array_equal(l["vertices"], (r["vertices"] @ REFLECTION)[:, [0, 2, 1]])
        for r, l in zip(right.findall("joint"), left.findall("joint")):
            np.testing.assert_allclose(vector(l.find("axis").get("xyz")),
                                       -REFLECTION @ vector(r.find("axis").get("xyz")), atol=1e-12)
            self.assertEqual(r.find("limit").attrib, l.find("limit").attrib)

    def test_upper_hinge_does_not_move_motor_or_shank(self):
        zero = copy.deepcopy(self.config)
        for side in zero["sides"]:
            zero["sides"][side]["upper_hinge_deg"] = 0
        for bend in (0, 30, 70):
            update(self.model, self.data, self.devices, zero, bend)
            pos, rot = self.data.xpos.copy(), self.data.xmat.copy()
            update(self.model, self.data, self.devices, self.config, bend)
            for side in ("R", "L"):
                for link in LINKS[1:]:
                    body = self.model.body(f"hinge_{side}_{link}").id
                    np.testing.assert_allclose(self.data.xpos[body], pos[body], atol=1e-12)
                    np.testing.assert_allclose(self.data.xmat[body], rot[body], atol=1e-12)
                body = self.model.body(f"hinge_{side}_base_link").id
                self.assertGreater(np.linalg.norm(self.data.xmat[body] - rot[body]), .001)

    def test_calibrated_motor_axis_is_coaxial_with_human_knee(self):
        for bend in (0, 30, 70, 90):
            report = update(self.model, self.data, self.devices, self.config, bend)
            for side in ("R", "L"):
                self.assertLess(report[side]["knee_axis_line_distance_mm"], 1e-6)
                self.assertLess(report[side]["knee_axis_angle_deg"], 1e-4)

    def test_lower_hinge_adjustment_does_not_move_motor(self):
        baseline = copy.deepcopy(self.config)
        for mount in baseline["sides"].values():
            mount["lower_hinge_deg"] = 0
        update(self.model, self.data, self.devices, baseline, 0)
        before_pos, before_rot = self.data.xpos.copy(), self.data.xmat.copy()
        update(self.model, self.data, self.devices, self.config, 0)
        for side in ("R", "L"):
            for link in ("motor_housing_link", "motor_output_link"):
                body = self.model.body(f"hinge_{side}_{link}").id
                np.testing.assert_allclose(self.data.xpos[body], before_pos[body], atol=1e-12)
                np.testing.assert_allclose(self.data.xmat[body], before_rot[body], atol=1e-12)
            body = self.model.body(f"hinge_{side}_shank_link").id
            self.assertGreater(np.linalg.norm(self.data.xmat[body] - before_rot[body]), .1)

    def test_bilateral_meshes_are_mirrored_throughout_motion(self):
        for bend in (0, 30, 70):
            update(self.model, self.data, self.devices, self.config, bend)
            for link in LINKS:
                r = mesh_geometry(self.model, self.data, self.model.geom(f"hinge_R_{link}_visual_0").id)[0]
                l = mesh_geometry(self.model, self.data, self.model.geom(f"hinge_L_{link}_visual_0").id)[0]
                # Compilation can reorder mesh vertices; compare nearest sets.
                from scipy.spatial import cKDTree
                self.assertLess(cKDTree(l).query(r * [1, -1, 1])[0].max(), 1e-6)

    def test_thigh_clearance_matches_five_millimeter_target(self):
        update(self.model, self.data, self.devices, self.config, 0)
        for side in ("R", "L"):
            first = mesh_geometry(self.model, self.data, self.model.geom(f"femur_{side.lower()}_geom_1").id)
            second = mesh_geometry(self.model, self.data, self.model.geom(f"hinge_{side}_base_link_visual_0").id)
            self.assertEqual(surface_crossings(first, second), 0)
            self.assertAlmostEqual(surface_distance(first, second) * 1000, 5, delta=.05)

    def test_import_retains_three_joints_and_preview_has_no_device_physics(self):
        for dm, _, _ in self.devices.values():
            self.assertEqual(dm.njnt, 3)
            self.assertEqual(dm.nu, 0)
        self.assertEqual(self.model.nmocap, 8)
        for side in ("R", "L"):
            for link in LINKS:
                body = self.model.body(f"hinge_{side}_{link}").id
                self.assertEqual(self.model.body_mass[body], 0)


if __name__ == "__main__":
    import unittest
    unittest.main()
