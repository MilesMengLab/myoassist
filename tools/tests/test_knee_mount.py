"""Geometric checks for the visual-only mounting calibration tool."""

import copy
from pathlib import Path
import tempfile
from unittest import TestCase

import numpy as np
from scipy.spatial import cKDTree

from tools.preview_knee_mount import DEFAULT_CONFIG, build, load_config, rotation, update


class KneeMountTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.model, cls.data, cls.devices = build(Path(cls.temp.name))
        cls.config = load_config(DEFAULT_CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_neutral_axes_are_coaxial(self):
        report = update(self.model, self.data, self.devices, self.config, 0)
        for side in ("R", "L"):
            self.assertLess(report[side]["axis_line_distance_mm"], 1e-6)
            self.assertLess(report[side]["axis_angle_deg"], 1e-4)

    def test_both_urdf_hinges_flex_positive_seventy_degrees(self):
        report = update(self.model, self.data, self.devices, self.config, 70)
        for side in ("R", "L"):
            self.assertAlmostEqual(report[side]["device_angle_deg"], 70)
            self.assertTrue(report[side]["within_urdf_joint_range"])

    def test_mount_offset_is_measured(self):
        config = copy.deepcopy(self.config)
        config["sides"]["R"]["translation_m"][0] += 0.02
        report = update(self.model, self.data, self.devices, config, 0)
        self.assertAlmostEqual(report["R"]["axis_line_distance_mm"], 20)

    def test_thigh_mesh_clearance_is_one_millimeter(self):
        from tools.check_knee_mount_clearance import mesh_geometry, surface_distance, surface_crossings

        calibrated = copy.deepcopy(self.config)
        # Check the measured candidate independently of the active user mount.
        calibrated["sides"]["R"]["translation_m"][2] = 0.06735332265807957
        update(self.model, self.data, self.devices, calibrated, 0)
        first = mesh_geometry(self.model, self.data, self.model.geom("femur_r_geom_1").id)
        second = mesh_geometry(self.model, self.data, self.model.geom("knee_R_upper_visual").id)
        self.assertEqual(surface_crossings(first, second), 0)
        self.assertAlmostEqual(surface_distance(first, second) * 1000, 1, places=2)

    def test_no_device_actuation_or_collisions(self):
        self.assertEqual(self.model.nu, 22)
        self.assertEqual(self.model.nmocap, 4)
        for side in ("R", "L"):
            for part in ("upper", "lower"):
                g = self.model.geom(f"knee_{side}_{part}_visual").id
                self.assertEqual(self.model.geom_contype[g], 0)
                self.assertEqual(self.model.geom_conaffinity[g], 0)
                b = self.model.body(f"knee_{side}_{part}").id
                self.assertEqual(self.model.body_mass[b], 0)

    def test_rpy_uses_fixed_axes(self):
        np.testing.assert_allclose(rotation([90, 0, 0]) @ [0, 1, 0], [0, 0, 1], atol=1e-12)

    def test_lower_offset_moves_only_lower_brace(self):
        baseline = copy.deepcopy(self.config)
        changed = copy.deepcopy(self.config)
        for side in ("R", "L"):
            baseline["sides"][side]["lower_offset_device_m"] = [0, 0, 0]
            changed["sides"][side]["lower_offset_device_m"] = [0, 0, .01]
        for bend in (0, 70):
            update(self.model, self.data, self.devices, baseline, bend)
            before = self.data.xpos.copy()
            update(self.model, self.data, self.devices, changed, bend)
            for side in ("R", "L"):
                upper = self.model.body(f"knee_{side}_upper").id
                lower = self.model.body(f"knee_{side}_lower").id
                np.testing.assert_array_equal(self.data.xpos[upper], before[upper])
                self.assertAlmostEqual(np.linalg.norm(self.data.xpos[lower] - before[lower]), .01)

    def test_default_keeps_motor_connector_assembly(self):
        for side in ("R", "L"):
            np.testing.assert_array_equal(self.config["sides"][side]["lower_offset_device_m"], [0, 0, 0])

    def test_left_geometry_mirrors_right_at_neutral_and_flexion(self):
        model, data = self.model, self.data
        for bend in (0, 70):
            update(model, data, self.devices, self.config, bend)
            for part in ("upper", "lower"):
                with self.subTest(bend=bend, part=part):
                    points = []
                    for side in ("R", "L"):
                        geom = model.geom(f"knee_{side}_{part}_visual").id
                        mesh = model.geom_dataid[geom]
                        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                        local = model.mesh_vert[first:first + count]
                        points.append(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
                    reflected = points[0][::max(1, len(points[0]) // 1000)] * [1, -1, 1]
                    distances = cKDTree(points[1]).query(reflected)[0]
                    self.assertLess(float(distances.max()), 1e-6)
