"""Import the exported knee URDFs into MuJoCo, preserving visual and inertial data.

This is a standalone device preview, not a human-device training environment.
"""

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[1]


def convert(source: Path, output: Path) -> dict:
    source = source.resolve()
    package = source.parent.parent
    tree = ET.parse(source)
    robot = tree.getroot()
    for mesh in robot.findall(".//mesh"):
        name = mesh.get("filename", "")
        prefix = f"package://{package.name}/"
        if name.startswith(prefix):
            path = package / name[len(prefix):]
        elif name.startswith("package://"):
            raise ValueError(f"Unknown ROS package in mesh URI: {name}")
        else:
            path = source.parent / name
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path.resolve()))

    extension = robot.find("mujoco")
    if extension is None:
        extension = ET.SubElement(robot, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    compiler.set("strippath", "false")

    output.mkdir(parents=True, exist_ok=True)
    cleaned = output / f"{source.stem}_resolved.urdf"
    tree.write(cleaned, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(cleaned))
    target = output / f"{source.stem}.xml"
    mujoco.mj_saveLastXML(str(target), model)
    # URDF effort limits are metadata, not automatically created MuJoCo motors.
    joints = []
    for joint in robot.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        axis = joint.find("axis")
        limits = joint.find("limit")
        joints.append({
            "name": joint.get("name"), "type": joint.get("type"),
            "axis": axis.get("xyz") if axis is not None else "1 0 0",
            "effort_limit": float(limits.get("effort")) if limits is not None else None,
        })
    report = {
        "source": str(source), "mjcf": str(target.resolve()),
        "bodies_excluding_world": model.nbody - 1,
        "joint_count": model.njnt, "actuator_count": model.nu,
        "total_mass_kg": float(model.body_mass.sum()),
        "joint_name": joints[0]["name"] if joints else None,
        "axis": joints[0]["axis"] if joints else None,
        "joint_range_rad": model.jnt_range.tolist(),
        "urdf_effort_limit": joints[0]["effort_limit"] if joints else None,
        "joints": joints,
        "status": "standalone device; human attachment and motor configuration required",
    }
    (output / f"{source.stem}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def preview(path: Path):
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("Standalone passive device preview. No human, policy or motor is loaded.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = max(0.7, model.stat.extent * 1.5)
        while viewer.is_running():
            start = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.monotonic() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT / "urdf/knee_exo_hinge")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--view", choices=["L", "R"], help="Open one converted device in MuJoCo")
    args = parser.parse_args()
    output = args.output or args.package / "mujoco"
    reports = {}
    for side in ("L", "R"):
        candidates = [args.package / "urdf" / f"{args.package.name}_{suffix}.urdf"
                      for suffix in (side.lower(), side)]
        source = next((p for p in candidates if p.is_file()), candidates[0])
        report = convert(source, output)
        reports[side] = report
        print(json.dumps(report, indent=2))
    if args.view:
        preview(Path(reports[args.view]["mjcf"]))


if __name__ == "__main__":
    main()
