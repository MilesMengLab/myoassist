"""Render Hippo_L1 with prescribed reference kinematics (no learned policy)."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from myoassist_utils.compose import compose_env_model


def main():
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
    renderer = mujoco.Renderer(model, height=720, width=960)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    with imageio.get_writer(out / "hippo_reference.mp4", fps=30) as writer:
        for frame in range(180):
            i = int(frame * rate / 30)
            data.qpos[:] = initial
            for addr, key in mapping:
                data.qpos[addr] = series[key][i]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = data.xpos[pelvis] if pelvis >= 0 else (data.qpos[0], 0, 0.9)
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            writer.append_data(pixels)
            if frame == 30:
                imageio.imwrite(out / "hippo_reference.png", pixels)
    renderer.close()
    print("Reference kinematics preview, no trained policy:", out / "hippo_reference.mp4")


if __name__ == "__main__":
    main()
