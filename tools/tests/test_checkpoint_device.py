"""Checkpoint evaluation must respect the configured device, including on GPU hosts."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from rl_train.envs.environment_handler import EnvironmentHandler


class CheckpointDeviceTest(TestCase):
    def test_evaluation_checkpoint_uses_configured_device(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), patch("stable_baselines3.PPO.load") as load:
                config = SimpleNamespace(
                    env_params=SimpleNamespace(env_id="myoAssistLegImitationExo-v0"),
                    ppo_params=SimpleNamespace(device=device),
                )
                env = object()

                result = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path="checkpoint")

                self.assertIs(result, load.return_value)
                self.assertEqual(load.call_args.kwargs["device"], device)
                self.assertIs(load.call_args.kwargs["env"], env)
