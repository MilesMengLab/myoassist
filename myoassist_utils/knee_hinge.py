"""Names and reset synchronization for the articulated local knee device."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "urdf/knee_exo_hinge"
DEFAULT_INTERFACE = PACKAGE / "training_interface.json"
ENCODER_SENSOR_NAMES = ("right_knee_encoder_pos", "right_knee_encoder_vel")


def joint_name(side, kind="knee"):
    return f"exo_{kind}_{side.lower()}"


def load_interface(path=DEFAULT_INTERFACE):
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    result["urdf"] = {s: path.parent / p for s, p in result["urdf"].items()}
    for key in ("mounting", "imu_mounts"):
        result[key] = path.parent / result[key]
    return result


def synchronize_reference_state(sim):
    """Match encoder pose/velocity to a newly initialized human reference.

    Applies only to models carrying our articulated joint names. Existing official
    and historical rigid-device environments keep their original reset behavior.
    """
    for side in ("r", "l"):
        try:
            motor = sim.model.joint(joint_name(side))
        except (KeyError, ValueError):
            continue
        human = sim.data.joint(f"knee_angle_{side}")
        sim.data.joint(joint_name(side)).qpos[0] = -human.qpos[0]
        sim.data.joint(joint_name(side)).qvel[0] = -human.qvel[0]
        for kind in ("upper_hinge", "lower_hinge"):
            name = joint_name(side, kind)
            joint = sim.model.joint(name)
            address = int(joint.qposadr[0])
            sim.data.joint(name).qpos[0] = sim.model.qpos0[address]
            sim.data.joint(name).qvel[0] = 0
