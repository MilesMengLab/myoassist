"""Exercise entry-point argument registration without starting training or a GUI."""

import argparse
from contextlib import chdir
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch


class ArgumentsCaptured(Exception):
    pass


class TrainCliTest(TestCase):
    def test_play_model_path_selects_matching_session_and_weights(self):
        from rl_train.envs.environment_handler import EnvironmentHandler

        root = Path(__file__).resolve().parents[2]
        source_config = root / "rl_train/train/train_configs/local_knee_exo/imitation_22_knee_exo_imu26.json"
        with TemporaryDirectory() as tmp, chdir(tmp):
            session = Path("session with spaces")
            checkpoint = session / "trained_models/model_123.zip"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            (session / "session_config.json").write_text(source_config.read_text())

            def capture_model(config, env):
                self.assertEqual(config.env_params.prev_trained_policy_path, str(checkpoint.resolve()))
                self.assertEqual(config.ppo_params.device, "cpu")
                self.assertFalse(config.env_params.flag_random_ref_index)
                raise ArgumentsCaptured

            for model_path in (str(checkpoint), str(checkpoint.resolve())):
                with (
                    self.subTest(model_path=model_path),
                    patch.object(sys, "argv", [
                        "run_train.py", "--play_model_path", model_path,
                        "--config.ppo_params.device", "cpu",
                        "--no-config.env_params.flag_random_ref_index",
                    ]),
                    patch.object(EnvironmentHandler, "create_environment") as create_env,
                    patch.object(EnvironmentHandler, "get_stable_baselines3_model", side_effect=capture_model),
                    self.assertRaises(ArgumentsCaptured),
                ):
                    runpy.run_path(str(root / "rl_train/run_train.py"), run_name="__main__")
                self.assertTrue(create_env.call_args.kwargs["is_evaluate_mode"])

    def test_imitation_reference_start_flag_is_registered(self):
        root = Path(__file__).resolve().parents[2]
        config = root / "rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json"
        original_parse = argparse.ArgumentParser.parse_args

        def capture(parser, *args, **kwargs):
            parsed = original_parse(parser, *args, **kwargs)
            self.assertIs(getattr(parsed, "config.env_params.flag_random_ref_index"), False)
            self.assertTrue(parsed.flag_realtime_evaluate)
            self.assertEqual(parsed.play_seconds, 120)
            raise ArgumentsCaptured

        argv = [
            "run_train.py", "--config_file_path", str(config),
            "--flag_realtime_evaluate", "--play_seconds", "120",
            "--no-config.env_params.flag_random_ref_index",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(argparse.ArgumentParser, "parse_args", capture),
            self.assertRaises(ArgumentsCaptured),
        ):
            runpy.run_path(str(root / "rl_train/run_train.py"), run_name="__main__")
