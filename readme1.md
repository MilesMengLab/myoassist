# 新版同轴合页膝外骨骼

在仓库根目录 `/home/meng/myoassist` 执行。输入接口为
`urdf/knee_exo_hinge/training_interface.json`；安装参数为同目录的
`mounting_preview.json`，右/左 URDF 在 `urdf/` 子目录。

## 安装预览

```bash
.mrl_venv/bin/python -m tools.preview_knee_hinge_mount
.mrl_venv/bin/python -m tools.preview_knee_hinge_mount --animate
```

## CPU 训练

```bash
.mrl_venv/bin/python -m tools.build_knee_training_model
.mrl_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/local_knee_exo/imitation_22_knee_exo_imu26.json \
  --config.env_params.num_envs 32 \
  --config.total_timesteps 30000000 \
  --config.ppo_params.device cpu \
  --no-flag_rendering
```

生成模型：`urdf/knee_exo_hinge/mujoco/training/human_knee_exo.xml`。
新模型保留每腿两个被动合页和一个电机膝轴，通过数值绑缚约束连接人体。
外骨骼输入仍为 26 维（四个 IMU＋右侧设备编码器），完整仿真观测改为 70 维。
需新训练权重；旧版 68 维检查点不能直接用于这个训练配置。

## Play：只填一次新模型路径

```bash
read -r -p "新合页模型路径（含.zip）: " knee_model_path
.mrl_venv/bin/python -m rl_train.run_train \
  --play_model_path "$knee_model_path" \
  --config.ppo_params.device cpu \
  --play_seconds 120 \
  --no-config.env_params.flag_random_ref_index
```

VS Code 的膝关节 Train 会自动重建当前模型；Play 输入一次对应新训练的 `.zip` 路径。
官方髋关节参考预览仍可使用：

```bash
.mrl_venv/bin/python -m tools.preview_hip_reference --view
```
