"""Build the local ideal knee-assistance model using calibrated visual mounts.

Device bodies are rigidly grafted to femur/tibia; there is no second hinge,
closed-chain constraint, strap compliance, or motor dynamics. Device collision
masks match the installed official Hippo model (contype=conaffinity=1).
"""

import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from myoassist_utils.compose import _absolutize_files
from tools.preview_knee_mount import DEFAULT_CONFIG, ROOT, build, load_config, quaternion, update
from tools.preview_knee_mount import rotation
from myoassist_utils.knee_imu import IMU_NAMES, INPUT_NAMES, INPUT_UNITS


DEFAULT_OUTPUT = ROOT / "urdf/knee_exo/mujoco/training/human_knee_exo.xml"


def build_training_model(mounting=DEFAULT_CONFIG, output=DEFAULT_OUTPUT, torque_nm=20.0,
                         imu_mounts=ROOT / "urdf/knee_exo/imu_mounts.json"):
    output = Path(output).resolve()
    if not np.isfinite(torque_nm) or torque_nm <= 0:
        raise ValueError("torque_nm must be positive and finite")
    config = load_config(mounting)
    if not all(m["enabled"] for m in config["sides"].values()):
        raise ValueError("Training configuration expects both devices enabled")
    model, data, devices = build(output.parent / "source_preview")
    alignment = update(model, data, devices, config, 0)
    root = ET.parse(output.parent / "source_preview/human_knee_mount.xml").getroot()
    _absolutize_files(root, output.parent / "source_preview")
    visual = root.find("visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    # The existing gait evaluator exports 1920x1080 videos.
    global_visual.set("offwidth", "1920")
    global_visual.set("offheight", "1080")
    world = root.find("worldbody")
    floor = world.find("geom[@name='mount_preview_floor']")
    # The source mounting preview has a non-colliding display floor. Training
    # needs a physical support surface, just like the official flat terrain.
    floor.set("contype", "1")
    floor.set("conaffinity", "1")
    human_mass = float(model.body_mass.sum())
    # Overlay bodies have zero mass; add the exported URDF inertial data to
    # fixed child bodies without changing existing human inertias.
    added_mass = 0.0
    for side in ("R", "L"):
        imported = ET.parse(output.parent / f"source_preview/imported/knee_exo_{side}.xml").getroot()
        upper = imported.find("worldbody/body")
        lower = upper.find("body")
        mount = config["sides"][side]
        for part, source, parent_name in (
            ("upper", upper, mount["parent_body"]),
            ("lower", lower, "tibia_" + side.lower()),
        ):
            name = f"knee_{side}_{part}"
            overlay = root.find(f"worldbody/body[@name='{name}']")
            parent_xml = root.find(f".//body[@name='{parent_name}']")
            parent = model.body(parent_name).id
            device = model.body(name).id
            parent_rotation = data.xmat[parent].reshape(3, 3)
            pos = parent_rotation.T @ (data.xpos[device] - data.xpos[parent])
            rot = parent_rotation.T @ data.xmat[device].reshape(3, 3)
            overlay.attrib.pop("mocap")
            overlay.set("pos", " ".join(map(str, pos)))
            overlay.set("quat", " ".join(map(str, quaternion(rot))))
            inertial = copy.deepcopy(source.find("inertial"))
            if inertial is None:
                raise ValueError(f"Missing URDF inertia: {side} {part}")
            added_mass += float(inertial.get("mass"))
            overlay.insert(0, inertial)
            for geom in overlay.findall("geom"):
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
            for site in list(overlay.findall("site")):
                overlay.remove(site)
            world.remove(overlay)
            parent_xml.append(overlay)
        axis_marker = root.find(f".//site[@name='human_axis_{side}']")
        for parent in root.iter():
            if axis_marker in list(parent):
                parent.remove(axis_marker)
                break
    actuator = root.find("actuator")
    imu_config = json.loads(Path(imu_mounts).read_text())
    entries = imu_config["sensors"]
    if [e["name"] for e in entries] != list(IMU_NAMES):
        raise ValueError(f"IMU order must be {IMU_NAMES}")
    sensors = root.find("sensor")
    for entry in entries:
        body = root.find(f".//body[@name='{entry['body']}']")
        if body is None:
            raise ValueError(f"Unknown IMU parent body: {entry['body']}")
        pos = np.asarray(entry["pos_m"], dtype=float)
        rpy = np.asarray(entry["rpy_deg"], dtype=float)
        if pos.shape != (3,) or rpy.shape != (3,) or not np.isfinite([pos, rpy]).all():
            raise ValueError(f"Invalid IMU mounting transform: {entry['name']}")
        site = f"imu_{entry['name']}_site"
        ET.SubElement(body, "site", name=site, pos=" ".join(map(str, pos)),
                      quat=" ".join(map(str, quaternion(rotation(rpy)))),
                      size="0.008", group="3", rgba="0 1 1 1")
        ET.SubElement(sensors, "accelerometer", name=f"imu_{entry['name']}_acc", site=site)
        ET.SubElement(sensors, "gyro", name=f"imu_{entry['name']}_gyro", site=site)
    for side in ("R", "L"):
        # Human knee q decreases in flexion. Positive normalized command
        # therefore produces negative generalized torque (flexion) on both legs.
        ET.SubElement(actuator, "general", name=f"Exo_{side}",
                      joint=f"knee_angle_{side.lower()}", ctrllimited="true",
                      ctrlrange="-1 1", forcelimited="true", forcerange=f"{-torque_nm} {torque_nm}",
                      gear="-1", gaintype="fixed", gainprm=str(torque_nm),
                      biastype="none", dyntype="none")
    for key in root.findall("keyframe/key"):
        if key.get("ctrl"):
            key.set("ctrl", key.get("ctrl") + " 0 0")
    root.set("model", "myolegs22_local_knee_exo_ideal")
    root.insert(0, ET.Comment(" Generated ideal knee assistance model. Device collision masks 1/1. "
                              "Not a closed-chain or motor-dynamics model. "))
    ET.indent(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(output))
    if compiled.nu != 24 or compiled.nq != model.nq or compiled.nmocap != 0:
        raise ValueError("Unexpected actuator, joint, or mocap layout")
    if not np.isclose(compiled.body_mass.sum(), human_mass + added_mass):
        raise ValueError("Device mass not preserved")
    state = mujoco.MjData(compiled)
    mujoco.mj_resetDataKeyframe(compiled, state, 0)
    mujoco.mj_forward(compiled, state)
    device_contacts = []
    for contact in state.contact:
        names = [compiled.geom(int(g)).name or "" for g in (contact.geom1, contact.geom2)]
        if any(name.startswith("knee_") for name in names):
            device_contacts.append({"geoms": names, "distance_mm": float(contact.dist * 1000)})
    report = {
        "model_path": str(output), "mounting_config": config,
        "human_mass_kg": human_mass, "device_mass_kg": added_mass,
        "combined_mass_kg": float(compiled.body_mass.sum()),
        "muscles": 22, "device_actuators": 2, "torque_limit_nm": torque_nm,
        "command_sign": "positive=human knee flexion, negative=extension",
        "assumptions": ["fixed upper-to-femur/lower-to-tibia attachments",
                        "device/ground collision masks 1/1, matching official Hippo; no strap compliance",
                        "ideal torque source; URDF velocity metadata is not enforced",
                        "human knee limits retained; device URDF limits not imposed"],
        "neutral_alignment": alignment,
        "collision_masks": {"device_contype": 1, "device_conaffinity": 1,
                            "floor_contype": 1, "floor_conaffinity": 1},
        "neutral_contact_count": state.ncon,
        "neutral_device_contacts": device_contacts,
        "imu_mounts": imu_config,
        "exo_input_names": INPUT_NAMES,
        "exo_input_units": INPUT_UNITS,
        "acceleration_convention": "site-frame specific force including gravity; no gravity removal or normalization",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounting", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--torque-nm", type=float, default=20, help="Ideal torque cap per knee, default from URDF effort")
    parser.add_argument("--imu-mounts", type=Path, default=ROOT / "urdf/knee_exo/imu_mounts.json")
    args = parser.parse_args()
    build_training_model(args.mounting, args.output, args.torque_nm, args.imu_mounts)


if __name__ == "__main__":
    main()
