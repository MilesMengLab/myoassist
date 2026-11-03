"""Random phase selection must never change the configured speed bounds."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from rl_train.envs.myoassist_leg_base import MyoAssistLegBase


class VelocityInitializationTest(TestCase):
    def test_random_modes_preserve_speed_bounds_across_resets(self):
        env = object.__new__(MyoAssistLegBase)
        env.sim = SimpleNamespace(data=SimpleNamespace(time=0.0))
        env._min_target_velocity = env._max_target_velocity = 1.25
        env._min_target_velocity_period = 2
        env._max_target_velocity_period = 10
        for mode in MyoAssistLegBase.VelocityMode:
            with self.subTest(mode=mode), patch(
                "rl_train.envs.myoassist_leg_base.random.choice", return_value=mode
            ), patch("rl_train.envs.myoassist_leg_base.random.uniform", side_effect=lambda a, b: (a + b) / 2):
                for _ in range(3):
                    env._change_mode_and_target_velocity_randomly()
                    self.assertEqual(env._min_target_velocity, 1.25)
                    self.assertEqual(env._max_target_velocity, 1.25)
                    self.assertEqual(env._target_velocity, 1.25)
