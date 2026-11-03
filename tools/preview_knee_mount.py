"""Visual-only human/knee-device mounting calibration; no policy or physics stepping."""

import argparse
import copy
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from assist_sim.registry import resolve
from assist_sim.utils import export_combined_xml
from tools.import_knee_urdf import convert


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "urdf/knee_exo/mounting_preview.json"
DEFAULT_OUTPUT = ROOT / "urdf/knee_exo/mujoco/mounting_preview"


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


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config["sides"]) != {"L", "R"}:
        raise ValueError("Provide both L and R mounting entries (enabled may be false).")
    for side, mount in config["sides"].items():
        if mount["parent_body"] != f"femur_{side.lower()}" or mount["human_joint"] != f"knee_angle_{side.lower()}":
            raise ValueError(f"{side}: this preview expects the corresponding femur and knee joint")
        for name, size in (("translation_m", 3), ("rotation_rpy_deg", 3), ("rgba", 4)):
            values = np.asarray(mount[name], dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"{side}.{name}: expected {size} finite numbers")
        offset = np.asarray(mount.get("lower_offset_device_m", [0, 0, 0]), dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError(f"{side}.lower_offset_device_m: expected three finite numbers")
        if not np.isfinite(mount["joint_zero_deg"]):
            raise ValueError(f"{side}.joint_zero_deg must be finite")
        if np.any(np.asarray(mount["rgba"]) < 0) or np.any(np.asarray(mount["rgba"]) > 1):
            raise ValueError(f"{side}.rgba must be in [0, 1]")
    if not np.isfinite(config["preview_flexion_deg"]):
        raise ValueError("preview_flexion_deg must be finite")
    return config


def build(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # resolve returns an unmodified human spec and a device config path; we use
    # only the human. No Hippo device is composed or attached here.
    human, _ = resolve("myolegs22", "Hippo_L1")
    root = ET.fromstring(human.to_xml())
    # Use assist_sim's reload-safe exporter for composed human defaults.
    compiler = root.find("compiler")
    export_combined_xml(
        human, str(output / "human_source.xml"),
        mesh_dirs=[(Path(human.modelfiledir or "."), compiler.get("meshdir", ""))],
    )
    root = ET.parse(output / "human_source.xml").getroot()
    asset = root.find("asset")
    world = root.find("worldbody")
    ET.SubElement(asset, "texture", name="mount_floor_tex", type="2d", builtin="checker",
                  rgb1="0.8 0.82 0.84", rgb2="0.7 0.72 0.74", width="128", height="128")
    ET.SubElement(asset, "material", name="mount_floor_mat", texture="mount_floor_tex",
                  texrepeat="3 3", reflectance="0")
    ET.SubElement(world, "geom", name="mount_preview_floor", type="plane", size="0 0 0.05",
                  material="mount_floor_mat", contype="0", conaffinity="0")
    device_info = {}
    for side in ("R", "L"):
        source = ROOT / f"urdf/knee_exo/urdf/knee_exo_{side}.urdf"
        report = convert(source, output / "imported")
        device = ET.parse(report["mjcf"]).getroot()
        for mesh in device.findall("asset/mesh"):
            entry = copy.deepcopy(mesh)
            entry.set("name", f"knee_{side}_{mesh.get('name')}")
            asset.append(entry)
        upper = device.find("worldbody/body")
        lower = upper.find("body")
        joint = lower.find("joint")
        device_info[side] = {
            "joint_pos": np.fromstring(joint.get("pos", "0 0 0"), sep=" "),
            "joint_axis": np.fromstring(joint.get("axis", "0 0 1"), sep=" "),
            "child_pos": np.fromstring(lower.get("pos", "0 0 0"), sep=" "),
            "child_rotation": np.eye(3),
            "range": np.fromstring(joint.get("range"), sep=" "),
        }
        if "quat" in lower.attrib:
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, np.fromstring(lower.get("quat"), sep=" "))
            device_info[side]["child_rotation"] = mat.reshape(3, 3)
        for part, source_body in (("upper", upper), ("lower", lower)):
            body = ET.SubElement(world, "body", name=f"knee_{side}_{part}", mocap="true")
            # Only URDF visual geoms. No collision mesh duplicates or inertial
            # contribution: these mocap bodies are overlays, not dynamics.
            for geom in source_body.findall("geom"):
                if geom.get("group") != "1":
                    continue
                entry = copy.deepcopy(geom)
                entry.set("mesh", f"knee_{side}_{geom.get('mesh')}")
                entry.set("name", f"knee_{side}_{part}_visual")
                entry.set("contype", "0")
                entry.set("conaffinity", "0")
                entry.set("density", "0")
                body.append(entry)
            if part == "upper":
                # Show the actual imported joint frame, not a guessed COM.
                pos = device_info[side]["child_pos"] + device_info[side]["child_rotation"] @ device_info[side]["joint_pos"]
                axis = device_info[side]["child_rotation"] @ device_info[side]["joint_axis"]
                ends = np.concatenate([pos - 0.07 * axis, pos + 0.07 * axis])
                ET.SubElement(body, "site", name=f"device_axis_{side}", type="capsule",
                              fromto=" ".join(map(str, ends)), size="0.003", rgba="1 0.8 0 1")
        human_joint = root.find(f".//joint[@name='knee_angle_{side.lower()}']")
        parent = next(b for b in root.findall(".//body") if human_joint in list(b))
        axis = np.fromstring(human_joint.get("axis", "0 0 1"), sep=" ")
        pos = np.fromstring(human_joint.get("pos", "0 0 0"), sep=" ")
        ends = np.concatenate([pos - 0.1 * axis, pos + 0.1 * axis])
        ET.SubElement(parent, "site", name=f"human_axis_{side}", type="capsule",
                      fromto=" ".join(map(str, ends)), size="0.002", rgba="0.1 0.9 0.2 1")
    target = output / "human_knee_mount.xml"
    from myoassist_utils.scene_colors import style_xml
    style_xml(root)
    ET.indent(root)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(target))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, device_info


def update(model, data, devices, config, flexion_deg):
    for mount in config["sides"].values():
        joint = model.joint(mount["human_joint"])
        data.qpos[joint.qposadr[0]] = -np.deg2rad(flexion_deg)
    mujoco.mj_forward(model, data)
    report = {}
    for side, mount in config["sides"].items():
        parent = model.body(mount["parent_body"]).id
        j = model.joint(mount["human_joint"]).id
        parent_rot = data.xmat[parent].reshape(3, 3)
        rot = parent_rot @ rotation(mount["rotation_rpy_deg"])
        pos = data.xpos[parent] + parent_rot @ np.asarray(mount["translation_m"])
        device = devices[side]
        child_rot = rot @ device["child_rotation"]
        child_pos = pos + rot @ device["child_pos"]
        pivot = child_pos + child_rot @ device["joint_pos"]
        axis = child_rot @ device["joint_axis"]
        alignment = float(np.clip(np.dot(data.xaxis[j], axis), -1, 1))
        angle = np.copysign(1.0, alignment) * data.qpos[model.jnt_qposadr[j]] + np.deg2rad(mount["joint_zero_deg"])
        lower_rot = child_rot @ axis_rotation(device["joint_axis"], angle)
        lower_pos = pivot - lower_rot @ device["joint_pos"]
        # Independent lower-brace fit adjustment, expressed in its local CAD
        # frame. The upper brace and shared mount parameters are untouched.
        lower_pos += lower_rot @ np.asarray(mount.get("lower_offset_device_m", [0, 0, 0]))
        for part, p, r in (("upper", pos, rot), ("lower", lower_pos, lower_rot)):
            body = model.body(f"knee_{side}_{part}").id
            mocap = model.body_mocapid[body]
            data.mocap_pos[mocap] = p
            data.mocap_quat[mocap] = quaternion(r)
            geom = model.geom(f"knee_{side}_{part}_visual").id
            model.geom_rgba[geom] = mount["rgba"]
            if not mount["enabled"]:
                model.geom_rgba[geom, 3] = 0
        delta = pivot - data.xanchor[j]
        perpendicular = delta - np.dot(delta, data.xaxis[j]) * data.xaxis[j]
        report[side] = {
            "axis_line_distance_mm": float(np.linalg.norm(perpendicular) * 1000),
            "axis_angle_deg": float(np.rad2deg(np.arccos(abs(alignment)))),
            "device_angle_deg": float(np.rad2deg(angle)),
            "within_urdf_joint_range": bool(device["range"][0] - 1e-6 <= angle <= device["range"][1] + 1e-6),
        }
    mujoco.mj_forward(model, data)
    return report


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render", action="store_true", help="Save three images instead of opening a window")
    parser.add_argument("--animate", action="store_true", help="Prescribed 0–70 degree knee bend, no policy")
    args = parser.parse_args()
    config = load_config(args.config)
    model, data, devices = build(args.output)
    report = update(model, data, devices, config, config["preview_flexion_deg"])
    print("VISUAL CALIBRATION ONLY: no physics, device inertia, motor, or trained policy.")
    print("Green = human knee axis; yellow = device axis; orange = right; blue = left.")
    print(json.dumps(report, indent=2))
    (args.output / "alignment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.render:
        snapshot(model, data, args.output)
        print("Images:", args.output)
        return
    import mujoco.viewer

    started = time.monotonic()
    stamp = args.config.stat().st_mtime_ns
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.85]
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -8
        print("Edit and save the mounting JSON to reload positions and rotations live. Ctrl+C exits.")
        while viewer.is_running():
            reloaded = False
            new_stamp = args.config.stat().st_mtime_ns
            if new_stamp != stamp:
                try:
                    config = load_config(args.config)
                    print("Reloaded mounting config")
                    reloaded = True
                except (ValueError, KeyError) as exc:
                    print("Keeping previous valid config:", exc)
                stamp = new_stamp
            bend = 35 * (1 - np.cos((time.monotonic() - started) * np.pi / 3)) if args.animate else config["preview_flexion_deg"]
            with viewer.lock():
                report = update(model, data, devices, config, bend)
            if reloaded:
                print(json.dumps(report, indent=2))
                (args.output / "alignment_report.json").write_text(json.dumps(report, indent=2) + "\n")
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
