"""Triangle surface audit must detect crossing and separate surfaces."""

from unittest import TestCase

import numpy as np

from tools.check_knee_mount_clearance import surface_crossings, surface_distance


class MeshClearanceTest(TestCase):
    def test_crossing_and_separated_triangles(self):
        faces = np.array([[0, 1, 2]])
        first = (np.array([[-1., -1, 0], [1, -1, 0], [0, 1, 0]]), faces)
        second = np.array([[0., 0, -1], [0, 0, 1], [1, 0, 0.5]])
        self.assertGreater(surface_crossings(first, (second, faces)), 0)
        self.assertEqual(surface_crossings(first, (second + [0, 0, 3], faces)), 0)

    def test_face_clearance_in_meters(self):
        faces = np.array([[0, 1, 2]])
        large = np.array([[-1., -1, 0], [1, -1, 0], [0, 1, 0]])
        small = np.array([[-0.1, -0.1, .001], [.1, -.1, .001], [0, .1, .001]])
        self.assertAlmostEqual(surface_distance((large, faces), (small, faces)), .001, places=10)

    def test_interior_edge_clearance(self):
        faces = np.array([[0, 1, 2]])
        first = np.array([[-1., 0, 0], [1, 0, 0], [0, -1, -1]])
        second = np.array([[0., -1, .01], [0, 1, .01], [0, 0, 1]])
        self.assertAlmostEqual(surface_distance((first, faces), (second, faces)), .01, places=10)
