"""Interactive playback timing and camera behavior without opening a GUI."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import mujoco

from rl_train import run_train


class PlayTest(TestCase):
    def run_play(self, seconds=None, terminated=False, truncated=False, window_open=True):
        config = SimpleNamespace(
            env_params=SimpleNamespace(control_framerate=30),
            evaluate_param_list=[{
                "num_timesteps": 200, "cam_distance": 2.5,
                "min_target_velocity": 1.25, "max_target_velocity": 1.25,
                "velocity_mode": "UNIFORM", "target_velocity_period": 2,
            }],
        )
        env = MagicMock()
        env.reset.return_value = ("obs", {})
        env.step.return_value = ("obs", 0.0, terminated, truncated, {})
        env.sim.renderer._window.is_running.return_value = window_open
        model = MagicMock()
        model.predict.return_value = ("action", None)
        with (
            patch.object(run_train.EnvironmentHandler, "create_environment", return_value=env),
            patch.object(run_train.EnvironmentHandler, "get_stable_baselines3_model", return_value=model),
            patch.object(run_train.EnvironmentHandler, "updateconfig_from_model_policy"),
            patch("mujoco.mj_name2id", return_value=3),
            patch("myoassist_utils.scene_colors.style_model", return_value=[]),
        ):
            run_train.ppo_evaluate_with_rendering(config, play_seconds=seconds)
        env.close.assert_called_once()
        return env

    def test_duration_and_tracking(self):
        env = self.run_play(seconds=2)
        self.assertEqual(env.step.call_count, 60)
        camera = env.sim.renderer._window.cam
        self.assertEqual(camera.type, mujoco.mjtCamera.mjCAMERA_TRACKING)
        self.assertEqual(camera.trackbodyid, 3)
        self.assertEqual(camera.distance, 2.5)
        settings = env.set_target_velocity_mode_manually.call_args.kwargs
        self.assertEqual(settings["initial_target_velocity"], 1.25)
        self.assertEqual(settings["max_target_velocity"], 1.25)

    def test_default_duration(self):
        self.assertEqual(self.run_play().step.call_count, 200)

    def test_termination_does_not_restart_continuous_playback(self):
        env = self.run_play(seconds=0.1, terminated=True)
        self.assertEqual(env.reset.call_count, 1)

    def test_time_limit_resets(self):
        env = self.run_play(seconds=0.1, truncated=True)
        self.assertEqual(env.reset.call_count, 4)

    def test_closed_window_stops_playback(self):
        self.assertEqual(self.run_play(window_open=False).step.call_count, 0)

