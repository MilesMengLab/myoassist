"""Build the aligned four-link/three-joint device into the human RL model.

Thigh cuffs attach to femurs; full URDF chains retain passive hinges and driven
knees. Numerical welds connect shank cuffs to tibias to close the kinematic loops.
"""

import argparse
import copy
from itertools import combinations
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from myoassist_utils.compose import _absolutize_files
from myoassist_utils.knee_hinge import (
    DEFAULT_INTERFACE, ENCODER_SENSOR_NAMES, PACKAGE, joint_name, load_interface,
)
from myoassist_utils.knee_imu import IMU_NAMES, INPUT_NAMES, INPUT_UNITS
from tools.preview_knee_hinge_mount import (
    LINKS, body_pose, build, load_config, quaternion, rotation, update,
)

DEFAULT_OUTPUT = PACKAGE / "mujoco/training/human_knee_exo.xml"


def numbers(values):
    return " ".join(f"{v:.16g}" for v in np.asarray(values).reshape(-1))


def build_training_model(mounting=None, output=DEFAULT_OUTPUT, torque_nm=20.0,
                         imu_mounts=None, interface=DEFAULT_INTERFACE):
    settings = load_interface(interface)
    mounting = mounting or settings["mounting"]
    imu_mounts = imu_mounts or settings["imu_mounts"]
    config, dynamics = load_config(mounting), settings["dynamics"]
    output = Path(output).resolve()
    if not np.isfinite(torque_nm) or torque_nm <= 0:
        raise ValueError("torque_nm must be positive and finite")
    model, data, devices = build(output.parent / "source_preview", interface)
    alignment = update(model, data, devices, config, 0)
    if any(r["knee_axis_angle_deg"] > .01 or r["knee_axis_line_distance_mm"] > .01
           for r in alignment.values()):
        raise ValueError("Align motor and human knee axes before building the training model")
    root = ET.parse(output.parent / "source_preview/human_knee_hinge_mount.xml").getroot()
    _absolutize_files(root, output.parent / "source_preview")
    world = root.find("worldbody")
    floor = world.find("geom[@name='mount_preview_floor']")
    floor.set("contype", "1")
    floor.set("conaffinity", "1")
    visual = root.find("visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1920")
    global_visual.set("offheight", "1080")
    equality, contact = root.find("equality"), root.find("contact")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    # Joint insertion changes tree ordering. Rebuild keyframes by joint name,
    # not by appending device values to the human qpos vector.
    old_keys = copy.deepcopy(root.find("keyframe"))
    if root.find("keyframe") is not None:
        root.remove(root.find("keyframe"))
    for parent in root.iter():
        for site in list(parent.findall("site")):
            if site.get("name", "").startswith(("human_knee_axis_", "hinge_axis_")):
                parent.remove(site)
    human_mass, added_mass = float(model.body_mass.sum()), 0.
    joint_refs = {}
    for side in ("R", "L"):
        source = ET.parse(output.parent / f"source_preview/imported/{settings['urdf'][side].stem}.xml").getroot()
        names = []
        for index, link in enumerate(LINKS):
            name = f"hinge_{side}_{link}"
            names.append(name)
            overlay = world.find(f"body[@name='{name}']")
            source_body = source.find(f".//body[@name='{link}']")
            parent_name = f"femur_{side.lower()}" if index == 0 else names[index - 1]
            parent_id, body_id = model.body(parent_name).id, model.body(name).id
            relative = np.linalg.inv(body_pose(data, parent_id)) @ body_pose(data, body_id)
            overlay.attrib.pop("mocap")
            overlay.set("pos", numbers(relative[:3, 3]))
            overlay.set("quat", numbers(quaternion(relative[:3, :3])))
            inertial = copy.deepcopy(source_body.find("inertial"))
            if inertial is None:
                raise ValueError(f"No inertial data for {side}.{link}")
            added_mass += float(inertial.get("mass"))
            overlay.insert(0, inertial)
            for geom in overlay.findall("geom"):
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
            if index:
                kind = ("upper_hinge", "knee", "lower_hinge")[index - 1]
                joint = copy.deepcopy(source_body.find("joint"))
                name_joint = joint_name(side, kind)
                joint.set("name", name_joint)
                ref = 0. if kind == "knee" else float(np.deg2rad(config["sides"][side][f"{kind}_deg"]))
                joint.set("ref", str(ref))
                joint.set("damping", str(dynamics["passive_hinge_damping"]))
                joint.set("armature", str(dynamics["joint_armature"]))
                joint.attrib.pop("actuatorfrcrange", None)
                joint.attrib.pop("actuatorfrclimited", None)
                joint_refs[name_joint] = ref
                overlay.insert(1, joint)
            world.remove(overlay)
            root.find(f".//body[@name='{parent_name}']").append(overlay)
        tibia = f"tibia_{side.lower()}"
        relative = np.linalg.inv(body_pose(data, model.body(tibia).id)) @ body_pose(data, model.body(names[-1]).id)
        ET.SubElement(equality, "weld", name=f"knee_cuff_{side.lower()}", body1=tibia, body2=names[-1],
                      relpose=numbers(np.r_[relative[:3, 3], quaternion(relative[:3, :3])]),
                      solref=numbers(dynamics["strap_weld_solref"]),
                      solimp=numbers(dynamics["strap_weld_solimp"]),
                      torquescale=str(dynamics["strap_weld_torquescale"]))
        # Joint interfaces overlap; the welded cuff's convex hull fills its bore.
        # Attachment constraints supply internal loads, so exclude connected CAD
        # pairs. Ground/opposite-leg collisions and mesh masks remain enabled.
        for first, second in combinations(names, 2):
            ET.SubElement(contact, "exclude", body1=first, body2=second)
        ET.SubElement(contact, "exclude", body1=tibia, body2=names[-1])
    sensors = root.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(root, "sensor")
    imu_config = json.loads(Path(imu_mounts).read_text())
    if [e["name"] for e in imu_config["sensors"]] != list(IMU_NAMES):
        raise ValueError(f"IMU order must be {IMU_NAMES}")
    for entry in imu_config["sensors"]:
        body = root.find(f".//body[@name='{entry['body']}']")
        if body is None:
            raise ValueError(f"Unknown IMU body: {entry['body']}")
        pos, rpy = np.asarray(entry["pos_m"], float), np.asarray(entry["rpy_deg"], float)
        if pos.shape != (3,) or rpy.shape != (3,) or not np.isfinite([pos, rpy]).all():
            raise ValueError(f"Invalid IMU mount: {entry['name']}")
        site = f"imu_{entry['name']}_site"
        ET.SubElement(body, "site", name=site, pos=numbers(pos), quat=numbers(quaternion(rotation(rpy))),
                      size=".008", group="3", rgba="0 1 1 1")
        ET.SubElement(sensors, "accelerometer", name=f"imu_{entry['name']}_acc", site=site)
        ET.SubElement(sensors, "gyro", name=f"imu_{entry['name']}_gyro", site=site)
    ET.SubElement(sensors, "jointpos", name=ENCODER_SENSOR_NAMES[0], joint=joint_name("r"))
    ET.SubElement(sensors, "jointvel", name=ENCODER_SENSOR_NAMES[1], joint=joint_name("r"))
    actuator = root.find("actuator")
    for side in ("R", "L"):
        ET.SubElement(actuator, "general", name=f"Exo_{side}", joint=joint_name(side),
                      ctrllimited="true", ctrlrange="-1 1", forcelimited="true",
                      forcerange=f"{-torque_nm} {torque_nm}", gear="1", gaintype="fixed",
                      gainprm=str(torque_nm), biastype="none", dyntype="none")
    root.set("model", "myolegs22_aligned_knee_hinge")
    ET.indent(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(output))
    if compiled.nu != 24 or compiled.nq != model.nq + 6 or compiled.nmocap:
        raise ValueError("Unexpected articulated device layout")
    keyframe = ET.SubElement(root, "keyframe")
    for index, old_key in enumerate(list(old_keys) if old_keys is not None else []):
        qpos, qvel = compiled.qpos0.copy(), np.zeros(compiled.nv)
        for old_joint in range(model.njnt):
            name = model.joint(old_joint).name
            new_joint = compiled.joint(name).id
            qpos[compiled.jnt_qposadr[new_joint]] = model.key_qpos[index, model.jnt_qposadr[old_joint]]
            qvel[compiled.jnt_dofadr[new_joint]] = model.key_qvel[index, model.jnt_dofadr[old_joint]]
        for side in ("r", "l"):
            motor, human = compiled.joint(joint_name(side)), compiled.joint(f"knee_angle_{side}")
            qpos[motor.qposadr[0]] = -qpos[human.qposadr[0]]
            qvel[motor.dofadr[0]] = -qvel[human.dofadr[0]]
        ET.SubElement(keyframe, "key", name=old_key.get("name"), qpos=numbers(qpos), qvel=numbers(qvel),
                      ctrl=numbers(np.r_[model.key_ctrl[index, :22], 0., 0.]))
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(output))
    state = mujoco.MjData(compiled)
    mujoco.mj_resetDataKeyframe(compiled, state, 0)
    mujoco.mj_forward(compiled, state)
    if not np.isclose(compiled.body_mass.sum(), human_mass + added_mass):
        raise ValueError("Device mass not preserved")
    device_contacts = []
    for c in state.contact:
        names = [compiled.geom(int(g)).name or "" for g in (c.geom1, c.geom2)]
        if any(n.startswith("hinge_") for n in names):
            device_contacts.append({"geoms": names, "distance_mm": float(c.dist * 1000)})
    report = {
        "model_path": str(output), "interface": str(Path(interface).resolve()),
        "urdf_sources": {s: str(p) for s, p in settings["urdf"].items()},
        "mounting_config": config, "neutral_alignment": alignment,
        "human_mass_kg": human_mass, "device_mass_kg": added_mass,
        "combined_mass_kg": float(compiled.body_mass.sum()),
        "device_joints": list(joint_refs), "joint_references_rad": joint_refs,
        "muscles": 22, "device_actuators": 2, "torque_limit_nm": torque_nm,
        "command_sign": "positive device encoder/torque = flexion; human knee coordinate is negative in flexion",
        "coupling": "thigh rigid mount, three real URDF joints, tibia-shank numerical weld",
        "dynamics": dynamics, "neutral_contact_count": state.ncon,
        "neutral_device_contacts": device_contacts,
        "collision_masks": {"device_contype": 1, "device_conaffinity": 1,
                            "floor_contype": 1, "floor_conaffinity": 1},
        "collision_exclusions": "connected parts within each device plus welded tibia/shank pair",
        "imu_mounts": imu_config, "exo_input_names": INPUT_NAMES, "exo_input_units": INPUT_UNITS,
        "encoder_sensors": ENCODER_SENSOR_NAMES,
        "acceleration_convention": "site-frame specific force; raw SI units",
        "assumptions": ["ideal torque motor; no electrical or bandwidth model",
                        "URDF knee/hinge joint limits retained",
                        "numerical weld is a cuff constraint, not skin/strap-pressure modeling"],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--mounting", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--torque-nm", type=float, default=20)
    parser.add_argument("--imu-mounts", type=Path)
    args = parser.parse_args()
    build_training_model(args.mounting, args.output, args.torque_nm, args.imu_mounts, args.interface)


if __name__ == "__main__":
    main()
