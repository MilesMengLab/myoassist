"""Compatibility entry point: use the aligned articulated knee preview."""

from tools.preview_knee_hinge_mount import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, ROOT, build, load_config, main, update,
)
from tools.knee_mount_utils import axis_rotation, quaternion, rotation, snapshot


if __name__ == "__main__":
    main()
