import os

# Cap the OpenMP-family thread pools before numpy or torch is imported -- they read
# these at import time, so this block has to stay above every other import.
#
# PPO's update, not the environment, dominates wall time here: at num_envs=64 a rollout
# spends ~5 s stepping MuJoCo and the rest in the update. Left at its default, torch
# fans every op out across all cores while the env workers compete for those same cores,
# and the networks ([64, 64]) are far too small for that to pay off. Measured on a
# 64-core / 128-thread box, imitation config, num_envs=64, steps/sec:
#
#     threads     1      4      8     16    default (64)
#     steps/s  2286   2731   2979   2809            819
#
# 8 is the peak and is used as the default; the win over the untuned default is ~3.6x.
# The right value is machine-dependent, so MYOASSIST_NUM_THREADS overrides it, and an
# OMP_NUM_THREADS you set yourself is left alone.
_MYOASSIST_THREADS = os.environ.get("MYOASSIST_NUM_THREADS", "8")
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, _MYOASSIST_THREADS)

import numpy as np  # noqa: E402  -- must follow the thread-pool caps above
import rl_train.train.train_configs.config as myoassist_config  # noqa: E402
import rl_train.utils.train_log_handler as train_log_handler  # noqa: E402
from rl_train.utils.data_types import DictionableDataclass  # noqa: E402
import json  # noqa: E402
from datetime import datetime  # noqa: E402
from rl_train.envs.environment_handler import EnvironmentHandler  # noqa: E402
import subprocess  # noqa: E402
import torch  # noqa: E402

# The env vars above are enough on Linux and Windows, where torch's CPU backend reads
# OMP_NUM_THREADS at import. macOS torch does not, so set it explicitly too; both paths
# then agree on the resolved value, whether it came from the default or an override.
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        # Provenance metadata only -- a run outside a git checkout still has to start.
        return {"commit": "unknown", "branch": "unknown"}


# Version information
VERSION = {
    "version": "0.3.0",  # MAJOR.MINOR.PATCH
    **get_git_info(),
}


def ppo_evaluate_with_rendering(config, play_seconds=None):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on=True, is_evaluate_mode=True)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

    evaluation = config.evaluate_param_list[0]
    min_velocity = evaluation["min_target_velocity"]
    max_velocity = evaluation["max_target_velocity"]
    env.set_target_velocity_mode_manually(
        mode=MyoAssistLegBase.VelocityMode[evaluation["velocity_mode"]],
        starting_phase=0,
        initial_target_velocity=(min_velocity + max_velocity) / 2,
        min_target_velocity=min_velocity,
        max_target_velocity=max_velocity,
        target_velocity_period=evaluation["target_velocity_period"],
    )
    obs, info = env.reset()
    import mujoco

    # Restore infinite floor display for older local knee scenes. Plane XY
    # size controls visualization only; keep archived XML and contacts intact.
    floor_id = mujoco.mj_name2id(env.sim.model.ptr, mujoco.mjtObj.mjOBJ_GEOM, "mount_preview_floor")
    if floor_id >= 0 and env.sim.model.ptr.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_PLANE:
        env.sim.model.ptr.geom_size[floor_id, :2] = 0
    from myoassist_utils.scene_colors import style_model
    changed_textures = style_model(env.sim.model.ptr)

    # The interactive viewer has its own camera; video camera settings do not
    # apply to it. Track the pelvis while allowing mouse rotation and zoom.
    env.sim.renderer.render_to_window()
    viewer = env.sim.renderer._window
    if viewer is not None:
        for texture_id in changed_textures:
            viewer.update_texture(texture_id)
    pelvis_id = mujoco.mj_name2id(env.sim.model.ptr, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if viewer is not None and pelvis_id >= 0:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis_id
            viewer.cam.distance = config.evaluate_param_list[0].get("cam_distance", 2.5)

    steps = config.evaluate_param_list[0]["num_timesteps"]
    if play_seconds is not None:
        steps = max(1, round(play_seconds * config.env_params.control_framerate))
    print(f"Play: {steps} steps, {steps / config.env_params.control_framerate:.1f} simulated seconds")
    try:
        for _ in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, truncated, info = env.step(action)
            # Preserve the original Play behavior: termination thresholds used
            # during training must not repeatedly restart a continuous replay.
            if truncated:
                obs, info = env.reset()
    finally:
        env.close()


def ppo_train_with_parameters(config, train_time_step, is_rendering_on, train_log_handler):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    if config.env_params.model_path:
        # Keep explicit custom models with each run so rebuilding a local
        # device later does not silently change playback of older checkpoints.
        from pathlib import Path

        source = config.env_params.model_path
        xml = source if source.lstrip().startswith("<") else Path(source).read_text()
        from myoassist_utils.compose import _absolutize_files
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)
        _absolutize_files(root, Path.cwd() if source.lstrip().startswith("<") else Path(source).resolve().parent)
        saved = Path(train_log_handler.log_dir).resolve() / "environment.xml"
        ET.ElementTree(root).write(saved, encoding="utf-8", xml_declaration=True)
        config.env_params.model_path = str(saved)

    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)

    session_config_dict["code_version"] = VERSION
    with open(os.path.join(train_log_handler.log_dir, "session_config.json"), "w", encoding="utf-8") as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)

    custom_callback = EnvironmentHandler.get_callback(config, train_log_handler)

    model.learn(
        reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1, callback=custom_callback, progress_bar=True
    )
    env.close()
    print("learning done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file_path", type=str, default="", help="path to train config file")
    parser.add_argument(
        "--play_model_path", type=str, default=None,
        help="Play a model .zip using session_config.json from its training directory",
    )
    parser.add_argument("--play_seconds", type=float, default=None, help="Play duration in simulated seconds")
    parser.add_argument(
        "--flag_rendering", type=bool, default=False, action=argparse.BooleanOptionalAction, help="rendering(True/False)"
    )
    parser.add_argument(
        "--flag_realtime_evaluate",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="realtime evaluate(True/False)",
    )

    args, unknown_args = parser.parse_known_args()
    play_model_path = None
    if args.play_model_path:
        from pathlib import Path

        play_model_path = Path(args.play_model_path).expanduser().resolve()
        if play_model_path.suffix != ".zip" or not play_model_path.is_file():
            parser.error(f"Model .zip file not found: {play_model_path}")
        session_config = play_model_path.parent.parent / "session_config.json"
        if not session_config.is_file():
            parser.error(f"Model's session_config.json not found: {session_config}")
        if args.config_file_path and Path(args.config_file_path).resolve() != session_config:
            parser.error("--config_file_path must match the model's session_config.json")
        args.config_file_path = str(session_config)
    if args.config_file_path is None:
        raise ValueError("config_file_path is required")

    default_config = EnvironmentHandler.get_session_config_from_path(
        args.config_file_path, myoassist_config.TrainSessionConfigBase
    )
    # Register subclass fields too (e.g. imitation's random reference start).
    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, config_type)
    DictionableDataclass.add_arguments(config, parser, prefix="config.")
    args = parser.parse_args()
    if args.play_seconds is not None and (not np.isfinite(args.play_seconds) or args.play_seconds <= 0):
        parser.error("--play_seconds must be a positive finite number")

    DictionableDataclass.set_from_args(config, args, prefix="config.")
    if play_model_path is not None:
        config.env_params.prev_trained_policy_path = str(play_model_path)
        args.flag_realtime_evaluate = True

    # Second-resolution timestamps collide when runs are launched together, and `exist_ok=True`
    # then silently hands two trainings the same directory: they interleave writes to
    # train_log.json and overwrite each other's checkpoints. That happened during the eight-device
    # sweep -- OpenExo_L1 and Humotech_L1 both landed on train_session_20260818-030709, OpenExo_L1
    # died reading a half-written log, and Humotech_L1 kept running with a corrupted one (33
    # checkpoints against 32 log entries). Claiming the directory with exist_ok=False and stepping
    # to the next free suffix makes a collision impossible rather than unlikely.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in range(100):
        candidate = os.path.join("rl_train", "results", f"train_session_{stamp}" + (f"_{suffix}" if suffix else ""))
        try:
            os.makedirs(candidate, exist_ok=False)
        except FileExistsError:
            continue
        log_dir = candidate
        break
    else:
        raise RuntimeError(f"could not claim a session directory for {stamp}: 100 suffixes already exist")
    print(f"Session directory: {log_dir}")
    train_log_handler = train_log_handler.TrainLogHandler(log_dir)

    if args.flag_realtime_evaluate:
        ppo_evaluate_with_rendering(config, play_seconds=args.play_seconds)
    else:
        ppo_train_with_parameters(
            config,
            train_time_step=config.total_timesteps,
            is_rendering_on=args.flag_rendering,
            train_log_handler=train_log_handler,
        )
