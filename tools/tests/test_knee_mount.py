"""Legacy preview commands must use the current aligned URDF implementation."""

from unittest import TestCase
from unittest.mock import patch
from tools import preview_knee_mount, preview_knee_hinge_mount
from tools import check_knee_mount_clearance, check_knee_hinge_clearance


class KneeMountCompatibilityTest(TestCase):
    def test_legacy_preview_uses_aligned_hinge_model(self):
        self.assertIs(preview_knee_mount.build, preview_knee_hinge_mount.build)
        self.assertIs(preview_knee_mount.main, preview_knee_hinge_mount.main)
        self.assertEqual(preview_knee_mount.DEFAULT_CONFIG, preview_knee_hinge_mount.DEFAULT_CONFIG)

    def test_legacy_clearance_command_delegates_to_current_audit(self):
        with patch.object(check_knee_hinge_clearance, "audit", return_value="report") as audit:
            self.assertEqual(check_knee_mount_clearance.audit(), "report")
            self.assertEqual(audit.call_args.args[0], preview_knee_hinge_mount.DEFAULT_CONFIG)
