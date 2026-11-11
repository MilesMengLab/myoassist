"""Hardware actor gets only four six-axis IMUs and right knee q/dq."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase

import mujoco
import numpy as np
import torch

from myoassist_utils.knee_imu import ACTOR_INDICES, IMU_NAMES, IMU_SENSOR_NAMES, INPUT_NAMES
from myoassist_utils.knee_hinge import ENCODER_SENSOR_NAMES
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from tools.build_knee_training_model import build_training_model


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "rl_train/train/train_configs/local_knee_exo/imitation_22_knee_exo_imu26.json"


class KneeImu26Test(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            path = build_training_model(output=Path(cls.temp.name) / "model.xml")
            c = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(CONFIG.read_text()))
            c.env_params.model_path = str(path)
            c.env_params.num_envs = 1
            c.ppo_params.batch_size = 32
            cls.env = EnvironmentHandler.create_environment(c, is_rendering_on=False)
            cls.agent = EnvironmentHandler.get_stable_baselines3_model(c, cls.env)
        cls.config = c

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.temp.cleanup()

    def test_observation_layout_and_raw_sensor_units(self):
        env = self.env
        self.assertEqual(env.observation_space.shape, (70,))
        self.assertEqual(env.action_space.shape, (24,))
        self.assertEqual(len(INPUT_NAMES), 26)
        obs, _ = env.reset()
        actual = obs[list(ACTOR_INDICES)]
        expected = np.concatenate([
            *[env.sim.data.sensor(name).data.copy() for name in IMU_SENSOR_NAMES],
            *[env.sim.data.sensor(name).data.copy() for name in ENCODER_SENSOR_NAMES],
        ])
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_actor_cannot_see_left_encoder_or_privileged_state(self):
        net = self.agent.policy.policy_network
        self.assertEqual(net.exo_policy_net[0].in_features, 26)
        self.assertEqual(net.exo_policy_net[-2].out_features, 2)
        self.assertEqual(self.config.ppo_params.mirror_coef, 0)
        obs = torch.randn(2, 70)
        modified = obs.clone()
        hidden = [i for i in range(70) if i not in ACTOR_INDICES]
        self.assertIn(4, hidden)  # privileged human left knee angle
        self.assertIn(5, hidden)  # privileged human right knee angle
        self.assertIn(12, hidden)  # left knee angular velocity
        modified[:, hidden] = torch.randn(2, len(hidden)) * 10
        with torch.no_grad():
            torch.testing.assert_close(net.forward_actor(obs)[:, 22:], net.forward_actor(modified)[:, 22:])
            mapped = net.network_index_handler.map_observation_to_network(obs, "exo_actor")
            torch.testing.assert_close(mapped, obs[:, list(ACTOR_INDICES)])

    def test_gyros_are_local_sensor_frame_angular_velocities(self):
        m = self.env.sim.model.ptr
        d = self.env.sim.data.ptr
        self.env.reset()
        d.qvel[:] = np.linspace(-0.5, 0.5, m.nv)
        mujoco.mj_forward(m, d)
        for name in IMU_NAMES:
            with self.subTest(imu=name):
                site = m.site(f"imu_{name}_site").id
                velocity = np.empty(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0)
                expected = d.site_xmat[site].reshape(3, 3).T @ velocity[:3]
                np.testing.assert_allclose(d.sensor(f"imu_{name}_gyro").data, expected, atol=1e-9)
                self.assertEqual(m.sensor(f"imu_{name}_acc").dim[0], 3)

    def test_random_reference_reset_synchronizes_actual_device_encoder(self):
        env = self.env
        previous = env._flag_random_ref_index
        try:
            env._flag_random_ref_index = True
            for _ in range(4):
                obs, _ = env.reset()
                self.assertTrue(np.isfinite(obs).all())
                self.assertAlmostEqual(float(env.sim.data.joint("exo_knee_r").qpos[0]),
                                       -float(env.sim.data.joint("knee_angle_r").qpos[0]))
                self.assertAlmostEqual(float(env.sim.data.joint("exo_knee_r").qvel[0]),
                                       -float(env.sim.data.joint("knee_angle_r").qvel[0]))
                np.testing.assert_allclose(obs[67:69], [env.sim.data.joint("exo_knee_r").qpos[0],
                                                      env.sim.data.joint("exo_knee_r").qvel[0]], rtol=1e-5)
        finally:
            env._flag_random_ref_index = previous
