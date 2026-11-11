"""Shared rotations and screenshots for knee installation tools."""

from pathlib import Path
import mujoco
import numpy as np


def rotation(rpy_deg):
    """URDF extrinsic XYZ / fixed-axis RPY: Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.deg2rad(rpy_deg)
    sx, sy, sz = np.sin([roll, pitch, yaw])
    cx, cy, cz = np.cos([roll, pitch, yaw])
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def quaternion(matrix):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.asarray(matrix).reshape(9))
    return result



def snapshot(model, data, output):
    import imageio.v2 as imageio

    model.vis.global_.offwidth = 1000
    model.vis.global_.offheight = 900
    renderer = mujoco.Renderer(model, height=900, width=1000)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.85]
    camera.distance = 2.8
    camera.elevation = -8
    try:
        for name, azimuth in (("front", 0), ("side", 90), ("oblique", 45)):
            camera.azimuth = azimuth
            renderer.update_scene(data, camera=camera)
            imageio.imwrite(Path(output) / f"{name}.png", renderer.render())
        camera.lookat[:] = [0, 0, 0.5]
        camera.distance = 1.25
        camera.azimuth = 45
        renderer.update_scene(data, camera=camera)
        imageio.imwrite(Path(output) / "knee_closeup.png", renderer.render())
    finally:
        renderer.close()


