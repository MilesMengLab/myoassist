# Knee exoskeleton import

From the MyoAssist repository root:

```bash
.mrl_venv/bin/python -m tools.import_knee_urdf
.mrl_venv/bin/python -m tools.import_knee_urdf --view R
.mrl_venv/bin/python -m tools.import_knee_urdf --view L
```

The importer resolves `package://knee_exo/` mesh paths, preserves visual meshes
and fixed bodies, and writes MJCF and an import report under `mujoco/`. Original
URDF/STL files are unchanged. Generated files use absolute mesh paths and are
ignored by Git: regenerate them after moving the repository to another machine.

The viewer is a **standalone passive device preview**. It does not load a human,
trained policy, ground, or motor. Its orientation is the original CAD frame;
the root body is anchored to the world. The hinge moves under gravity.

## Imported structure

Each side contains `base_link`, `shank_link`, and `knee_joint`:

| Property | Value |
| --- | --- |
| Base mass | 0.35343 kg |
| Shank mass | 0.19648 kg |
| Total mass | 0.54991 kg |
| Knee range | 0–1.9199 rad (about 110 degrees) |
| Right joint axis | local +Z |
| Left joint axis | local -Z |
| MuJoCo actuators after URDF import | 0 |

The URDF `effort=20` and `velocity=20` are exported metadata; importing them does
not create a MuJoCo motor or implement a motor velocity limiter. Confirm these
values against the real actuator before using them as training limits.

## Connecting to MyoAssist

A device-only XML cannot replace the existing human environment. Complete these
steps before training:

1. Confirm the CAD origin, knee axis, leg lengths, and left/right mounting frames.
   The left URDF is now generated from the right via CAD local X reflection;
   see the mirroring section below for mesh and inertial transformations.
2. Choose the attachment model. An ideal L1 model grafts the upper/lower device
   bodies onto human femur/tibia bodies and applies device torque to the human
   knee. A model retaining the device hinge needs an appropriate closed-chain
   attachment/constraint design and validation of joint-axis misalignment.
3. Add the actuator transmission, sign, torque limits, and collision exclusions.
   Do not duplicate a device hinge and a human joint without checking constraints.
4. Build a combined human + device + terrain MJCF. Use either an `assist_sim`
   device XML/YAML integration, or MyoAssist's `env_params.model_path` escape hatch
   with both `msk_key` and `device_key` set to null.
5. Create a knee policy config with the composed actuator count and knee
   observations; the Hippo hip policy is not a trained knee controller.
6. Validate neutral pose, knee flexion, gravity, contacts, and zero/positive/negative
   torque before launching training.

No human attachment transforms or knee training configuration have been guessed
by this importer.

## Human + device mounting preview

Current active preview: the user restored the **original 45 mm axial offsets**
with zero lower-only translation. This preserves the original assembly and may
show the original bone/brace intersections. The 67.3533 mm / 1 mm-clearance
calibration described below is a previous candidate, not the current setting.

```bash
# Neutral pose; edit/save mounting_preview.json to update the open window.
.mrl_venv/bin/python -m tools.preview_knee_mount

# Prescribed 0–70 degree bilateral knee flexion, for checking alignment.
.mrl_venv/bin/python -m tools.preview_knee_mount --animate

# Headless front/side/oblique/close-up screenshots, using EGL on Linux.
MUJOCO_GL=egl .mrl_venv/bin/python -m tools.preview_knee_mount --render
```

This is a **visual-only kinematic calibration** with mocap overlays. The original
URDF joint origins and meshes are preserved. There is no physics integration,
device inertia, device motor, learned controller, or human/device closed chain.
The exported XML is a scene asset for this tool; the Python updater places its
four device mocap overlays onto the current human pose.

Edit `mounting_preview.json`:

- `translation_m`: device base origin in the corresponding human femur frame,
  in **meters**. These are not global coordinates or URDF joint-origin edits.
- `rotation_rpy_deg`: device base orientation relative to the femur, in **degrees**,
  with URDF fixed-axis RPY convention (`Rz(yaw) Ry(pitch) Rx(roll)`).
- `joint_zero_deg`: device hinge zero offset, in degrees. Neutral export starts at 0.
- `lower_offset_device_m`: optional lower-only translation in local CAD meters.
  Keep `[0,0,0]` to preserve the original motor/connector assembly. A trial 24 mm
  shift reduced calf clearance but detached the motor connection, so it has been
  reverted. Do not use this field to correct cuff fit on an assembled device;
  revise only the cuff/brace geometry in CAD instead.
- `rgba`: device tint/opacity; right is orange and left is blue by default.
- `enabled`: show/hide one side's device meshes.
- `preview_flexion_deg`: static human knee flexion; ignored with `--animate`.

The initial mounting is **provisional**. It places each device pivot on the neutral
human knee-axis line with a 67.3533 mm axial displacement toward that leg's outside.
This is calibrated to approximately 1.000 mm minimum upper-brace/femur mesh surface
clearance at neutral pose. It replaces the original intersecting 45 mm offset
and the overly distant 85 mm trial. The complete device moves inward by 17.6467 mm
from that trial, bringing the shank brace closer while keeping upper/lower assembly
geometry consistent before the optional lower-only adjustment described above.
The lower-only trial is reverted: motor/connector relative positions follow the original URDF assembly.
Both devices move along the knee axis;
brace dimensions, original URDF joint origins, meshes and inertias are unchanged.
The model's local femur +Y points toward the hip; the CAD upper segment points
toward local -Y, so a 180 degree orientation adjustment is necessary. Right and
left rotations also account for the opposite URDF hinge-axis directions. These
are alignment candidates to inspect, not measured hardware mounting transforms.

Green rods show the human knee axes; yellow rods show the device knee axes.
`axis_line_distance_mm` measures perpendicular axis-line separation, not distance
between the two axis marker origins (a displacement along the axis is allowed).
`axis_angle_deg` treats parallel/antiparallel lines as aligned; the signed mapping
to the URDF hinge is checked separately via `within_urdf_joint_range`.

Calibration sequence:

1. Start neutral and compare front/side views. Confirm that the upper and lower
   braces face the right way and wrap around the correct leg.
2. Adjust the translation along the knee axis to set lateral clearance. Then
   adjust the other translations and rotations as required by the actual mount.
3. Inspect 0, 30, 60, and 70 degree flexion for axis orientation, knee clearance,
   shank-brace motion, and strap fit. This does not simulate soft tissue contact.
4. Confirm left/right mechanical geometry: the left URDF references real mirrored
   `base_link_L.STL` and `shank_link_L.STL`. The mounting preview does not apply
   any additional mesh reflection. The left mounting rotation maps this CAD
   reflection to the anatomical sagittal mirror. The hinge retains its -Z axis.
5. After visual approval, design the physical attachment, collision rules, inertia
   integration and actuator limits. Only then create a knee training environment.

This tool runs `mj_forward`, not `mj_step`. No falling or policy learning is
expected; animation is prescribed to inspect geometry rather than test balance.

Run `.mrl_venv/bin/python -m tools.check_knee_mount_clearance` to check actual
device-vs-human mesh surface crossings at 0, 30, 60 and 90 degree flexion,
independently of MuJoCo's collision filtering. Results are written to
`mujoco/mounting_preview/clearance_report.json`. These discrete-pose checks do not
certify all possible motions, anatomical skin clearance, muscle-line clearance,
or real strap fit. The model has no complete skin/soft-tissue envelope.

Rebuild with `python -m tools.build_knee_training_model` before new training.
Previously saved training-session XMLs retain their original mounting geometry.

## Generate the left URDF from the right

```bash
.mrl_venv/bin/python -m tools.mirror_knee_urdf
.mrl_venv/bin/python -m tools.import_knee_urdf
```

`knee_exo_R.urdf`, `base_link.STL`, and `shank_link.STL` are the source and are not
modified. The generator replaces `knee_exo_L.urdf` and writes `base_link_L.STL`
and `shank_link_L.STL`. Both visual and collision geometry reference these files
with positive scale, so importers do not need negative mesh-scale support.

The reflection is CAD-local `S=diag(-1,1,1)`. It changes mesh vertex X and reverses
triangle winding, transforms COM as `S*c`, and inertia as `S*I*S.T`. Origin
rotations transform as `S*R*S.T`. Revolute axes transform as `det(S)*S*a`, hence
right +Z becomes left -Z with the same positive angle limits. Mass, effort and
velocity metadata are preserved. This creates a mathematical mirror; it does not
certify the real left hardware's manufacturing geometry, mass or motor ratings.

After a new SolidWorks export of the right URDF, rerun these commands. Existing
CSV and ROS joint-name configuration files are not rewritten by the generator.
