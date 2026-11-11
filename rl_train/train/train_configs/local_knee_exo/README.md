# Aligned articulated knee-exoskeleton training

All local knee presets now load the aligned URDFs in `urdf/knee_exo_hinge`.
The historical `urdf/knee_exo` package is no longer a default input.

## Input interface and build

`urdf/knee_exo_hinge/training_interface.json` is the input manifest. Paths in it
are relative to that package and select:

- `urdf/knee_exo_hinge_r.urdf` and `urdf/knee_exo_hinge_l.urdf`;
- `mounting_preview.json`: calibrated installation and hinge reference angles;
- `imu_mounts.json`: four IMU sites;
- `dynamics`: added joint damping/armature and numerical cuff-weld parameters.

From the repository root:

```bash
.mrl_venv/bin/python -m tools.build_knee_training_model
.mrl_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/local_knee_exo/imitation_22_knee_exo_imu26.json \
  --config.env_params.num_envs 32 --config.ppo_params.device cpu --no-flag_rendering
```

The builder writes `urdf/knee_exo_hinge/mujoco/training/human_knee_exo.xml` and a
construction report beside it. Both local presets use that XML as `model_path`.
VS Code's **Train | 膝关节 | KneeExo Hinge IMU26 | CPU** entry rebuilds it using
**Build aligned knee hinge training model** before starting training.
Rebuild after changing any input. `--interface`, `--mounting`, `--imu-mounts`,
`--output` and `--torque-nm` allow explicit overrides.

## Model and coupling

The model retains all four URDF links and three revolute joints per leg:

```text
femur -> base_link -> passive upper hinge -> motor housing
      -> driven device knee -> motor output -> passive lower hinge -> shank
                                                                     |
                                                               cuff weld to tibia
```

- Device mass/inertia is preserved: **1.10216 kg** in total. The model has 45
  coordinates, six more than the previous rigid device, 22 muscle actuators and
  two motor actuators. No device mocap bodies remain in the training model.
- Upper/lower reference angles are approximately **1.2058361° / −8.25°**.
  Geometry at the first keyframe matches the calibrated preview. URDF knee
  limits (0–1.9199 rad) and passive hinge limits (approximately ±20°) are retained.
- Thigh cuffs are fixed to the femurs. A MuJoCo equality weld connects each shank
  cuff to the corresponding tibia, closing the kinematic loop. Upper/lower
  hinges have no actuators, and their movement responds to dynamics/constraints.
  Their freedom is consequently limited by human motion and the cuff constraints.
- The weld's `solref`, `solimp` and `torquescale` are numerical coupling choices,
  not measured skin or strap properties. They must not be interpreted as a
  validated soft-tissue, pressure or comfortable-fit model.
- Device and floor collision masks remain **1/1**. Explicit exclusions cover
  connected device parts within each leg and the bonded tibia/shank-cuff pair.
  This avoids internal hinge-interface contact and the solid convex hull filling
  a hollow cuff. Ground, opposite-leg and other external contacts remain enabled.
- `Exo_R/L` act on the actual `exo_knee_r/l` joints. Command [-1,1] produces
  [-20,20] N·m by default. Positive device angle and torque mean knee flexion;
  the human's `knee_angle_r/l` coordinate is negative in flexion. Loads transfer
  to the human through the device chain and cuff constraints.
- The motor is an ideal torque source. No electrical model, speed limiter,
  measured friction or actuator bandwidth is implemented.
- A reference reset initializes device encoder position/velocity consistently
  with human knee position/velocity and restores passive hinges to their fitting
  references. This synchronization is initialization only, not per-step forcing.
- Keyframes are remapped by joint name when the device joints are inserted.

See [MuJoCo's weld reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld)
for the numerical constraint definition.

## IMU26 inputs

The active preset remains `imitation_22_knee_exo_imu26.json`: CPU, 32 environments,
30 million steps, 30 Hz control / 1200 Hz physics. Its exo actor has exactly 26
inputs, in this order:

1. Left thigh ax, ay, az, gx, gy, gz.
2. Right thigh ax, ay, az, gx, gy, gz.
3. Left shank ax, ay, az, gx, gy, gz.
4. Right shank ax, ay, az, gx, gy, gz.
5. Actual right device encoder position and velocity (`exo_knee_r`).

Accelerations are local site-frame specific force in m/s², with gravity response;
gyros are local rad/s. Encoder angle/velocity use rad and rad/s, **positive in
flexion**. These are raw SI channels. See `urdf/knee_exo_hinge/imu26_input_schema.json`.
IMU locations currently use exported brace COMs with link-parallel axes as
provisional mounts. Replace them with measured sensor placement and calibration.
No sensor noise, bias, filtering, delay or dropout model is included.

The full simulation observation is now **70-dimensional**:
8 human qpos + 9 human qvel + 22 muscle activations + 4 foot forces + 24 IMU
channels + 2 device encoder channels + target speed. The exo actor reads indices
43–68 inclusive. Human actor/critic retain their 44 privileged inputs (0–42 and
69). Neither the left encoder, foot forces nor privileged human state enter the
exo actor. `mirror_coef=0` remains intentional for this asymmetric encoder input.

The earlier `imitation_22_knee_exo_ideal.json` preset is a 44-observation foot-sensor
baseline, but its **device geometry/dynamics also use the new hinge model**.

## Training, playback and interpretation

Old 68-observation checkpoints do not match the new 70-observation preset. Train
new weights rather than applying old weights directly to the new model. Historical
session configs still identify their own archived models; no old session is
silently migrated. Referenced mesh files remain external and must be preserved
for exact historical replay.

```bash
.mrl_venv/bin/python -m rl_train.run_train \
  --play_model_path rl_train/results/YOUR_NEW_SESSION/trained_models/model_STEPS.zip \
  --config.ppo_params.device cpu --play_seconds 120 \
  --no-config.env_params.flag_random_ref_index
```

Play accepts one full model path and loads that session's `session_config.json`.
The current XML is snapshotted into each new training session as `environment.xml`.

In the official evaluation plot, the human knee `qfrc_actuator` is now the direct
muscle contribution; the motor acts at a separate device joint and transfers load
through constraints. Do not label that human curve as muscle-plus-motor torque.
Device torque is separately recorded under `actuator_data.Exo_R/L`; the existing
Exo plot uses the human flexion-negative sign convention.

Short pipeline checks establish construction, torque direction/limits, encoder
reset, observation indexing, CPU PPO update, checkpoint loading and evaluation
compatibility. They do not establish convergence or assistance benefit. Use motor-
off baselines and repeated gait evaluations for that assessment.
