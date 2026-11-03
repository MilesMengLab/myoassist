# Local ideal knee-exoskeleton training

This is a custom configuration for this repository's `urdf/knee_exo` hardware,
derived from the official Hippo imitation preset. It is not an upstream preset.

## Build and train

From the repository root:

```bash
.mrl_venv/bin/python -m tools.build_knee_training_model
.mrl_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/local_knee_exo/imitation_22_knee_exo_imu26.json \
  --config.env_params.num_envs 32 --config.ppo_params.device cpu --no-flag_rendering
```

The builder reads the current right/left URDFs, their STL files, and
`urdf/knee_exo/mounting_preview.json`. It produces the model and a construction
report under `urdf/knee_exo/mujoco/training/` (ignored generated files). Rebuild
after any URDF, mesh, mounting, or torque-limit change. VS Code's custom knee
Train entry runs the builder as a pre-launch task.

## Model assumptions

- Human: myolegs22, 22 muscle controls and two ideal knee motors.
- The two URDF link masses/inertias per leg are preserved. Device mass totals
  1.09982 kg for both legs; rigid upper/lower bodies are children of human
  femur/tibia respectively. The independently articulated device hinge is omitted.
- Device meshes follow the rigid attachments. Device and floor collision masks
  are `contype=1, conaffinity=1`, matching the installed official Hippo device.
  MuJoCo's normal same-weld/parent filtering still applies. There are no strap
  forces, compliance, or closed-chain constraints; a collision mask does not
  create a realistic skin/brace contact interface. Mesh collision uses convex
  hulls, so hollow braces may need separate convex collision pieces for fidelity.
  The source visual-only preview still disables contact and does not run dynamics.
  The builder explicitly enables the display floor as a physical training floor;
  this corrects the initial version which inadvertently inherited its disabled
  masks from the mounting preview. Runs made before this correction do not
  represent the intended ground-contact task and should not be used as baselines.
- Per-knee torque limit defaults to 20 Nm from the URDF effort metadata; command
  [-1,1] maps to human knee generalized torque [20,-20] Nm. Positive command means
  flexion, negative means extension. Change the builder's `--torque-nm` parameter
  to use another validated ideal torque cap.
- No motor bandwidth, speed limiter, friction, or electrical model is implemented.
- Human joint limits are retained. The device URDF's 0–110 degree mechanical stops
  are **not** imposed as human knee limits in this ideal assistance model.
- Current mounting transforms are the provisional geometry used in the preview;
  this implementation does not certify hardware fit or biomechanical fidelity.
  Being visually close to a bone/line-muscle model does not establish skin
  clearance; human soft tissue and strap compression are not represented here.

## Training configuration

CPU, 32 parallel environments, 30 million total steps, 30 Hz control / 1200 Hz
physics; PPO/reward settings otherwise follow the official Hippo preset.
The active config is `imitation_22_knee_exo_imu26.json`. The original
`imitation_22_knee_exo_ideal.json` remains as the earlier foot-sensor baseline.

The hardware-facing actor has exactly 26 inputs:

1. Left thigh: ax, ay, az, gx, gy, gz.
2. Right thigh: ax, ay, az, gx, gy, gz.
3. Left shank: ax, ay, az, gx, gy, gz.
4. Right shank: ax, ay, az, gx, gy, gz.
5. Right knee angle and angular velocity.

Acceleration is **site-frame specific force** in m/s² (the accelerometer convention
with gravity response), not world-frame or gravity-removed linear acceleration.
Gyroscope values are site-frame rad/s. Right knee q/dq are rad and rad/s,
with negative angle for human knee flexion. These 26 channels are not normalized.
Hardware readings in g, degrees/s or degrees need SI conversion and axis/zero
calibration. See `urdf/knee_exo/imu26_input_schema.json` for the exact order.

`urdf/knee_exo/imu_mounts.json` specifies each sensor's position and orientation
relative to its brace body. Current sites use the brace COM and link axes as
**provisional** positions. Replace them with measured IMU mount locations and
orientations before relying on simulation-to-hardware transfer. Rebuild the model
after any change. Sensor noise, bias, filtering, latency and dropout are not yet
modeled; policy update frequency remains 30 Hz even if real sensors sample faster.

One exo network `[32,32]` receives these 26 channels and outputs right/left torque
commands in that order. It cannot read the left encoder, foot forces, muscles,
pelvis state or target speed. The full simulation observation is 68-dimensional:
8 qpos + 9 qvel + 22 muscle activations + 4 foot forces + 24 IMU channels + target
speed. Human actor and critic use the previous privileged 44 channels through
updated indexing; those networks are not part of the deployed hardware actor.

The old left/right shared-side actor is replaced by this two-output actor.
`mirror_coef=0` for IMU26: right-encoder-only input is asymmetric and the existing
mirror permutation does not support six-axis IMU sign/frame transformations.
Reusing the old mirror constraint would produce incorrect training. The two
outputs may therefore learn different functions; symmetry is not enforced.

This input change requires **new training**. Foot-sensor or Hippo checkpoints
cannot be loaded into this architecture. Existing session configurations remain
the source of truth for replaying their corresponding older checkpoints.

The environment uses the custom `model_path` entry with `msk_key` and `device_key`
null. Environment action-layout checks and mirror permutations accept this path.
Each training run snapshots the explicit XML to its own `environment.xml` and
stores that path in `session_config.json`. That freezes geometry/inertia/configured
torque in the XML; referenced mesh assets are still external files. Do not replace
those mesh files if older runs need exact visual/collision reproducibility.

## Play a knee session

```bash
.mrl_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/results/YOUR_KNEE_SESSION/session_config.json \
  --config.env_params.prev_trained_policy_path rl_train/results/YOUR_KNEE_SESSION/trained_models/model_STEPS \
  --config.ppo_params.device cpu --flag_realtime_evaluate --play_seconds 60 \
  --no-config.env_params.flag_random_ref_index
```

Select a newly trained **knee** session. Existing Hippo policies share tensor
dimensions but drive different joints and are not trained knee controllers.

Smoke checks validate mass, torque direction/cap, observation/action layout,
mirror mappings, short CPU PPO training, checkpoint loading, and periodic
evaluation/video export. These checks do not establish learning convergence or
effective assistance. Compare learned gait and muscle effort against appropriate
zero-assistance baselines before drawing research conclusions.
