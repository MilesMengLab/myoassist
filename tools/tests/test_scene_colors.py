"""Scene recoloring must not change contacts or dynamics of an archived model."""

import xml.etree.ElementTree as ET
from unittest import TestCase

import mujoco
import numpy as np

from myoassist_utils.scene_colors import BLUE, style_model, style_xml


class SceneColorsTest(TestCase):
    def test_bright_palette_and_physics_unchanged(self):
        xml = '''<mujoco><visual><headlight/></visual><asset>
        <texture type="skybox" builtin="flat" rgb1="1 1 1" width="32" height="32"/>
        <texture name="mount_floor_tex" type="2d" builtin="checker" width="32" height="32"/>
        <material name="mount_floor_mat" texture="mount_floor_tex"/>
        </asset><worldbody>
        <geom name="mount_preview_floor" type="plane" size="0 0 .05" material="mount_floor_mat"/>
        <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
        </worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        before = data.qacc.copy()
        contacts = [(c.geom1, c.geom2, c.dist) for c in data.contact]
        self.assertEqual(len(style_model(model)), 2)
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(before, data.qacc)
        self.assertEqual(contacts, [(c.geom1, c.geom2, c.dist) for c in data.contact])
        np.testing.assert_allclose(model.vis.rgba.contactforce, BLUE)
        np.testing.assert_allclose(model.vis.headlight.ambient, [.45, .45, .45])
        floor_mat = model.mat("mount_floor_mat").id
        np.testing.assert_array_equal(model.mat_texrepeat[floor_mat], [5, 5])
        self.assertAlmostEqual(float(model.mat_reflectance[floor_mat]), .2, places=6)
        root = ET.fromstring(xml)
        style_xml(root)
        styled = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        np.testing.assert_allclose(styled.vis.headlight.ambient, model.vis.headlight.ambient)
        self.assertEqual(styled.texture("mount_floor_tex").width[0], 300)
