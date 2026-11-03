"""Dynamics/layout regression checks for the local ideal knee device."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase

import mujoco
import numpy as np

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from tools.build_knee_training_model import build_training_model


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "rl_train/train/train_configs/local_knee_exo/imitation_22_knee_exo_ideal.json"


class KneeTrainingTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.path = build_training_model(output=Path(cls.temp.name) / "model.xml")
        cls.model = mujoco.MjModel.from_xml_path(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_mass_and_rigid_attachment_layout(self):
        m = self.model
        self.assertEqual(m.nu, 24)
        self.assertEqual(m.na, 22)
        self.assertEqual(m.nmocap, 0)
        self.assertEqual(m.nq, 39)
        mass = 0.0
        for side in ("R", "L"):
            for part, parent in (("upper", "femur"), ("lower", "tibia")):
                body = m.body(f"knee_{side}_{part}").id
                self.assertEqual(m.body_parentid[body], m.body(f"{parent}_{side.lower()}").id)
                self.assertEqual(m.body_jntnum[body], 0)
                self.assertTrue(np.all(m.body_inertia[body] > 0))
                mass += m.body_mass[body]
        self.assertAlmostEqual(mass, 1.09982)

    def test_device_and_floor_collisions_are_enabled(self):
        names = ["mount_preview_floor"] + [
            f"knee_{side}_{part}_visual" for side in ("R", "L") for part in ("upper", "lower")
        ]
        for name in names:
            with self.subTest(geom=name):
                geom = self.model.geom(name).id
                self.assertEqual(self.model.geom_contype[geom], 1)
                self.assertEqual(self.model.geom_conaffinity[geom], 1)

    def test_neutral_feet_have_ground_contacts(self):
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        floor = m.geom("mount_preview_floor").id
        foot_contacts = [
            contact for contact in d.contact
            if floor in (contact.geom1, contact.geom2)
            and "knee_" not in (m.geom(int(contact.geom1)).name or "")
            and "knee_" not in (m.geom(int(contact.geom2)).name or "")
        ]
        self.assertTrue(foot_contacts, "Training floor must provide support contacts")

    def test_signed_motor_torque_and_force_cap(self):
        m = self.model
        for side in ("R", "L"):
            for command in (-2.0, -1.0, 0.0, 1.0, 2.0):
                with self.subTest(side=side, command=command):
                    d = mujoco.MjData(m)
                    mujoco.mj_resetDataKeyframe(m, d, 0)
                    motor = m.actuator(f"Exo_{side}").id
                    joint = m.joint(f"knee_angle_{side.lower()}")
                    mujoco.mj_forward(m, d)
                    baseline = d.qfrc_actuator.copy()  # includes passive muscle forces
                    d.ctrl[motor] = command
                    mujoco.mj_forward(m, d)
                    expected = np.zeros(m.nv)
                    expected[joint.dofadr[0]] = -20 * np.clip(command, -1, 1)
                    np.testing.assert_allclose(d.qfrc_actuator - baseline, expected, atol=1e-8)
                    self.assertAlmostEqual(d.actuator_force[motor], 20 * np.clip(command, -1, 1))

    def test_config_layout_and_mirror_map_for_custom_xml(self):
        c = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(CONFIG.read_text()))
        c.env_params.model_path = str(self.path)
        EnvironmentHandler._validate_action_layout(str(self.path), c)
        obs, action, n_muscles = EnvironmentHandler._mirror_permutations(c)
        self.assertEqual(n_muscles, 22)
        self.assertEqual(len(obs), 44)
        self.assertEqual(len(action), 24)
        self.assertEqual(int(action[22]), 23)
        self.assertEqual(int(action[23]), 22)
        indices = c.policy_params.custom_policy_params.net_indexing_info
        self.assertEqual(indices["exo_actor_r"]["observation"][0]["index"][:4], [5, 4, 13, 12])
        self.assertEqual(indices["exo_actor_l"]["observation"][0]["index"][:4], [4, 5, 12, 13])
