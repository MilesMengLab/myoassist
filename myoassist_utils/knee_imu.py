"""Single source of truth for the hardware-facing knee policy's 26 raw inputs."""

IMU_NAMES = ("left_thigh", "right_thigh", "left_shank", "right_shank")
IMU_SENSOR_NAMES = tuple(f"imu_{part}_{kind}" for part in IMU_NAMES for kind in ("acc", "gyro"))
INPUT_NAMES = tuple(
    f"{part}.{kind}_{axis}"
    for part in IMU_NAMES for kind in ("acc", "gyro") for axis in ("x", "y", "z")
) + ("right_knee.angle", "right_knee.angular_velocity")
INPUT_UNITS = tuple(unit for _ in IMU_NAMES for unit in ("m/s^2",) * 3 + ("rad/s",) * 3) + ("rad", "rad/s")
# Full simulation observations: qpos 8, qvel 9, muscles 22, foot forces 4,
# IMU channels 24, actual device encoder 2, target speed 1 (70 total).
# Actor gets no privileged human knee coordinate or foot force.
ACTOR_INDICES = tuple(range(43, 69))
