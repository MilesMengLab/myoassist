"""Render Hippo_L1 with prescribed reference kinematics (no learned policy)."""
import os
import argparse
import time

from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from myoassist_utils.compose import compose_env_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="Open an interactive reference-motion preview")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "rl_train/results/hippo_reference_preview"
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(compose_env_model("myolegs22", "Hippo_L1"))
    data = mujoco.MjData(model)
    ref = np.load(root / "rl_train/reference_data/short_reference_gait.npz", allow_pickle=True)
    series = ref["series_data"].item()
    rate = ref["metadata"].item()["sample_rate"]
    mujoco.mj_resetDataKeyframe(model, data, 0) if model.nkey else mujoco.mj_resetData(model, data)
    initial = data.qpos.copy()
    mapping = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if "q_" + name in series and model.jnt_type[j] in (2, 3):
            mapping.append((model.jnt_qposadr[j], "q_" + name))
    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.azimuth = 90
    cam.elevation = -10
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    def set_pose(frame):
        i = int(frame * rate / 30)
        data.qpos[:] = initial
        for addr, key in mapping:
            data.qpos[addr] = series[key][i]
        mujoco.mj_forward(model, data)

    if args.view:
        from mujoco import viewer

        set_pose(0)
        with viewer.launch_passive(model, data) as window:
            window.cam.distance = 3.0
            window.cam.azimuth = 45
            window.cam.elevation = -10
            frame = 0
            print("Official Hippo_L1 + human: reference motion only, no policy or dynamics. Close window to exit.")
            while window.is_running():
                started = time.monotonic()
                with window.lock():
                    set_pose(frame % 180)
                    window.cam.lookat[:] = data.xpos[pelvis]
                window.sync()
                frame += 1
                time.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
        return
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
    renderer = mujoco.Renderer(model, height=720, width=960)
    with imageio.get_writer(out / "hippo_reference.mp4", fps=30) as writer:
        for frame in range(180):
            set_pose(frame)
            cam.lookat[:] = data.xpos[pelvis] if pelvis >= 0 else (data.qpos[0], 0, 0.9)
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            writer.append_data(pixels)
            if frame == 30:
                imageio.imwrite(out / "hippo_reference.png", pixels)
                cam.azimuth = 45
                renderer.update_scene(data, camera=cam)
                imageio.imwrite(out / "hippo_oblique.png", renderer.render())
                cam.azimuth = 90
    renderer.close()
    print("Reference kinematics preview, no trained policy:", out / "hippo_reference.mp4")


if __name__ == "__main__":
    main()
