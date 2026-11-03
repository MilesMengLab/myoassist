# 启动静态校准窗口：

cd /home/meng/myoassist
.mrl_venv/bin/python -m tools.preview_knee_mount

# 查看 0～70° 屈膝动画：

.mrl_venv/bin/python -m tools.preview_knee_mount --animate

.mrl_venv/bin/python -m tools.preview_hip_reference --view

# 训练命令：

cd /home/meng/myoassist

.mrl_venv/bin/python -m tools.build_knee_training_model &&
.mrl_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/local_knee_exo/imitation_22_knee_exo_ideal.json \
  --config.env_params.num_envs 32 \
  --config.total_timesteps 30000000 \
  --config.ppo_params.device cpu \
  --no-flag_rendering

# Play 命令：先按提示填写这次膝部训练的目录和权重名。

cd /home/meng/myoassist

read -r -p "训练目录（rl_train/results/train_session_时间）: " knee_session
read -r -p "权重名（model_步数，不含.zip）: " knee_checkpoint

.mrl_venv/bin/python -m rl_train.run_train \
  --config_file_path "$knee_session/session_config.json" \
  --config.env_params.prev_trained_policy_path "$knee_session/trained_models/$knee_checkpoint" \
  --config.ppo_params.device cpu \
  --flag_realtime_evaluate \
  --play_seconds 60 \
  --no-config.env_params.flag_random_ref_index
