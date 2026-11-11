"""Preview the articulated knee device, holding the motor/shank mount fixed.

The upper brace moves through the exported upper_hinge_joint, not an independent
mesh translation. Prescribed motion only; no training policy or physics stepping.
"""

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
from myoassist_utils.scene_colors import style_xml
from tools.import_knee_urdf import convert
from tools.knee_mount_utils import quaternion, rotation, snapshot
from myoassist_utils.knee_hinge import DEFAULT_INTERFACE, load_interface


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "urdf/knee_exo_hinge"
DEFAULT_CONFIG = load_interface()["mounting"]
DEFAULT_OUTPUT = PACKAGE / "mujoco/mounting_preview"
LINKS = ("base_link", "motor_housing_link", "motor_output_link", "shank_link")


def pose(rot, pos):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rot, pos
    return result


def body_pose(data, body):
    return pose(data.xmat[body].reshape(3, 3), data.xpos[body])


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    for side in ("R", "L"):
        mount = config["sides"][side]
        for key in ("upper_hinge_deg", "lower_hinge_deg"):
            if not np.isfinite(mount[key]) or abs(mount[key]) > np.rad2deg(.34907):
                raise ValueError(f"{side}.{key} is outside the exported hinge limits")
        for key in ("translation_m", "rotation_rpy_deg"):
            v = np.asarray(mount[key], float)
            if v.shape != (3,) or not np.isfinite(v).all():
                raise ValueError(f"Invalid {side}.{key}")
    return config


def build(output=DEFAULT_OUTPUT, interface=DEFAULT_INTERFACE):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    human, _ = resolve("myolegs22", "Hippo_L1")
    compiler = ET.fromstring(human.to_xml()).find("compiler")
    export_combined_xml(
        human, str(output / "human_source.xml"),
        mesh_dirs=[(Path(human.modelfiledir or "."), compiler.get("meshdir", ""))],
    )
    root = ET.parse(output / "human_source.xml").getroot()
    asset, world = root.find("asset"), root.find("worldbody")
    ET.SubElement(asset, "texture", name="mount_floor_tex", type="2d", builtin="checker",
                  rgb1="0.8 0.82 0.84", rgb2="0.7 0.72 0.74", width="128", height="128")
    ET.SubElement(asset, "material", name="mount_floor_mat", texture="mount_floor_tex",
                  texrepeat="3 3", reflectance="0")
    ET.SubElement(world, "geom", name="mount_preview_floor", type="plane", size="0 0 .05",
                  material="mount_floor_mat", contype="0", conaffinity="0")
    devices = {}
    sources = load_interface(interface)["urdf"]
    for side in ("R", "L"):
        human_joint = root.find(f".//joint[@name='knee_angle_{side.lower()}']")
        human_body = next(b for b in root.findall(".//body") if human_joint in list(b))
        human_point = np.fromstring(human_joint.get("pos", "0 0 0"), sep=" ")
        human_axis = np.fromstring(human_joint.get("axis", "0 0 1"), sep=" ")
        ET.SubElement(human_body, "site", name=f"human_knee_axis_{side}", type="capsule",
                      fromto=" ".join(map(str, np.r_[human_point - .10 * human_axis,
                                                    human_point + .10 * human_axis])),
                      size=".0015", rgba=".1 .9 .2 1")
        report = convert(sources[side], output / "imported")
        xml = ET.parse(report["mjcf"]).getroot()
        dm = mujoco.MjModel.from_xml_path(report["mjcf"])
        dd = mujoco.MjData(dm)
        mujoco.mj_forward(dm, dd)
        devices[side] = (dm, dd, body_pose(dd, dm.body("motor_housing_link").id).copy())
        for mesh in xml.findall("asset/mesh"):
            entry = copy.deepcopy(mesh)
            entry.set("name", f"hinge_{side}_{mesh.get('name')}")
            asset.append(entry)
        for link in LINKS:
            source = xml.find(f".//body[@name='{link}']")
            body = ET.SubElement(world, "body", name=f"hinge_{side}_{link}", mocap="true")
            for index, geom in enumerate(source.findall("geom[@group='1']")):
                entry = copy.deepcopy(geom)
                entry.set("mesh", f"hinge_{side}_{geom.get('mesh')}")
                entry.set("name", f"hinge_{side}_{link}_visual_{index}")
                entry.set("contype", "0")
                entry.set("conaffinity", "0")
                entry.set("density", "0")
                body.append(entry)
        for joint_name in ("upper_hinge_joint", "knee_joint", "lower_hinge_joint"):
            joint = dm.joint(joint_name).id
            child = dm.jnt_bodyid[joint]
            parent = dm.body_parentid[child]
            link = dm.body(parent).name
            t = body_pose(dd, parent)
            point = t[:3, :3].T @ (dd.xanchor[joint] - t[:3, 3])
            axis = t[:3, :3].T @ dd.xaxis[joint]
            ends = np.r_[point - .035 * axis, point + .035 * axis]
            ET.SubElement(world.find(f"body[@name='hinge_{side}_{link}']"), "site",
                          name=f"hinge_axis_{side}_{joint_name}", type="capsule",
                          fromto=" ".join(map(str, ends)), size=".002",
                          rgba="1 .8 0 1" if joint_name == "knee_joint" else "0 1 1 1")
    style_xml(root)
    ET.indent(root)
    target = output / "human_knee_hinge_mount.xml"
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(target))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, devices


def update(model, data, devices, config, flexion_deg=0):
    for side in ("R", "L"):
        joint = model.joint(f"knee_angle_{side.lower()}")
        data.qpos[joint.qposadr[0]] = -np.deg2rad(flexion_deg)
    mujoco.mj_forward(model, data)
    report = {}
    for side in ("R", "L"):
        mount = config["sides"][side]
        dm, dd, housing_zero = devices[side]
        parent = model.body(f"femur_{side.lower()}").id
        original_mount = body_pose(data, parent) @ pose(
            rotation(mount["rotation_rpy_deg"]), np.asarray(mount["translation_m"]),
        )
        housing_target = original_mount @ housing_zero
        dd.qpos[:] = 0
        mujoco.mj_forward(dm, dd)
        knee = dm.joint("knee_joint").id
        human_knee = model.joint(f"knee_angle_{side.lower()}").id
        alignment = float(np.dot(original_mount[:3, :3] @ dd.xaxis[knee], data.xaxis[human_knee]))
        dd.qpos[dm.joint("knee_joint").qposadr[0]] = np.copysign(1., alignment) * (-np.deg2rad(flexion_deg))
        dd.qpos[dm.joint("upper_hinge_joint").qposadr[0]] = np.deg2rad(mount["upper_hinge_deg"])
        dd.qpos[dm.joint("lower_hinge_joint").qposadr[0]] = np.deg2rad(mount["lower_hinge_deg"])
        mujoco.mj_forward(dm, dd)
        # Re-root at the motor housing: changing the upper hinge only moves the
        # base/thigh brace. Every downstream motor/output/shank frame is fixed.
        base_world = housing_target @ np.linalg.inv(body_pose(dd, dm.body("motor_housing_link").id))
        for link in LINKS:
            t = base_world @ body_pose(dd, dm.body(link).id)
            body = model.body(f"hinge_{side}_{link}").id
            mocap = model.body_mocapid[body]
            data.mocap_pos[mocap] = t[:3, 3]
            data.mocap_quat[mocap] = quaternion(t[:3, :3])
            for g in range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]):
                model.geom_rgba[g] = mount["rgba"]
        pivot = base_world[:3, :3] @ dd.xanchor[knee] + base_world[:3, 3]
        axis = base_world[:3, :3] @ dd.xaxis[knee]
        delta = pivot - data.xanchor[human_knee]
        perpendicular = delta - np.dot(delta, data.xaxis[human_knee]) * data.xaxis[human_knee]
        report[side] = {
            "upper_hinge_deg": mount["upper_hinge_deg"],
            "lower_hinge_deg": mount["lower_hinge_deg"],
            "knee_axis_line_distance_mm": float(np.linalg.norm(perpendicular) * 1000),
            "knee_axis_angle_deg": float(np.rad2deg(np.arccos(np.clip(abs(np.dot(axis, data.xaxis[human_knee])), 0, 1)))),
        }
    mujoco.mj_forward(model, data)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--animate", action="store_true", help="Prescribed knee bend, not a trained policy")
    args = parser.parse_args()
    args.config = args.config or load_interface(args.interface)["mounting"]
    config = load_config(args.config)
    model, data, devices = build(args.output, args.interface)
    alignment = update(model, data, devices, config, config["preview_flexion_deg"])
    print("Mounting preview only. Orange=right; blue=left; cyan=passive hinges; yellow=motor axis; green=human knee axis.")
    print(json.dumps(alignment, indent=2))
    if args.render:
        snapshot(model, data, args.output)
        print("Images:", args.output)
        return
    import mujoco.viewer

    started, stamp = time.monotonic(), args.config.stat().st_mtime_ns
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, .85]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 45, -8
        while viewer.is_running():
            current = args.config.stat().st_mtime_ns
            if current != stamp:
                try:
                    config = load_config(args.config)
                    print("Reloaded mounting config")
                except (ValueError, KeyError) as exc:
                    print("Keeping last valid config:", exc)
                stamp = current
            bend = 35 * (1 - np.cos((time.monotonic() - started) * np.pi / 3)) if args.animate else config["preview_flexion_deg"]
            with viewer.lock():
                update(model, data, devices, config, bend)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
