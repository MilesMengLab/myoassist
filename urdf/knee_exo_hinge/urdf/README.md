# 膝外骨骼 URDF 镜像与安装校准记录

记录日期：2026-11-04。当前几何安装已校准，并已接入 CPU 训练流程：保留真实三关节链，用数值绑缚约束连接人体。尚未验证新模型的训练收敛和辅助收益。

## 当前 URDF 接入接口

统一输入清单为 [training_interface.json](../training_interface.json)：其中 `urdf.R/L` 选择本目录的左右 URDF，`mounting` 选择已校准的安装参数，`imu_mounts` 选择四个 IMU 的安装信息，`dynamics` 设置附加关节阻尼和数值绑缚约束参数。
预览与训练构建器都通过该清单读取 URDF。

训练转换入口为仓库根目录的 `tools/build_knee_training_model.py`：

```bash
.mrl_venv/bin/python -m tools.build_knee_training_model
```

输出为 `urdf/knee_exo_hinge/mujoco/training/human_knee_exo.xml`，膝部训练 JSON 的 `env_params.model_path` 已全部切到该路径，VS Code 的膝部 Train 会自动先重建它。
预览、转换、镜像工具及旧预览命令的默认入口也已切到新合页包。

训练中每腿四个 link、三个关节均保留：大腿部件固定于股骨，上下合页为被动关节，电机驱动独立的 `exo_knee_r/l`，小腿部件通过 MuJoCo equality weld 绑定胫骨形成闭环。总设备质量为 **1.10216 kg**，电机仍为每膝 ±20 N·m 的理想力矩源。
数值 weld 不等于真实皮肤或绑带压力模型；内部连接件及绑定的小腿/胫骨对有碰撞排除，地面和外部碰撞继续启用。

26 维硬件输入的最后两项现在读取右侧**设备关节编码器**，屈膝为正，单位为 rad、rad/s。完整仿真观测从旧版 68 维改为 70 维，外骨骼网络仍只读取 26 维。新模型应重新训练，不能直接套用旧版 68 维权重。
各历史训练目录仍使用自己的模型快照，没有改写旧训练记录。
历史 XML 的网格引用仍是外部路径。为兼容旧策略 Play，保留
`urdf/knee_exo/meshes/base_link_L.STL` 和 `shank_link_L.STL` 等原版网格；
这两个左侧文件已从 Git 原始版本恢复。它们仅供旧训练回放，新训练仍读取新合页包。

完整训练与 Play 命令、观测布局和假设见 [训练说明](../../../rl_train/train/train_configs/local_knee_exo/README.md)。

本次接入验证：35 项相关检查通过；另以 2 个 CPU 并行环境完成 128 步短训练，模型保存、子进程自动评估／视频输出和检查点重新加载均通过。重新加载时观测为 70 维、动作为 24 维，计算设备为 CPU。该短测试仅确认流程可运行，不代表已完成新模型训练。

## URDF 文件处理

- 右腿：`knee_exo_hinge_r.urdf`。
- 左腿：`knee_exo_hinge_l.urdf`，由右腿镜像生成。
- 原始右腿导出文件备份：[knee_exo_hinge_r_export.urdf](../source/knee_exo_hinge_r_export.urdf)。
- 原导出文件重复定义了两个同名 `knee_joint`，连接同一对父子刚体。删除冗余的 `fixed` 项，保留 `revolute` 膝屈伸关节。
- 左腿以 CAD 局部 X=0 平面镜像：生成四个真实的 `../meshes/*_L.STL`，并同步镜像关节位置与朝向、旋转轴、质心及惯量；质量和关节限位保持一致。三角面绕序和法向也已处理。

保留的机构链为：

```text
base_link（大腿绑缚及连接杆）
  └─ upper_hinge_joint（上方被动合页）
      └─ motor_housing_link（电机固定侧）
          └─ knee_joint（电机膝屈伸轴）
              └─ motor_output_link（输出连接件）
                  └─ lower_hinge_joint（下方被动合页）
                      └─ shank_link（小腿绑缚及连接杆）
```

上下合页的导出限位均约为 ±20°。预览使用完整机构的正向运动学，没有独立平移、缩放零件或拆开连接。

## 电机轴校准

右侧 URDF 的膝关节轴原始定义为 `(0, 0.10771, 0.99418)`，该向量位于 URDF 关节坐标系中，不能直接当作人体世界坐标方向。

校准前，电机轴心到人体膝轴的距离约为 **0.00047 mm**，但轴向夹角约为 **6.1836°**。
在站立姿态下，从腿内侧沿轴走向外侧，电机轴是外侧高、内侧低；几乎没有前后方向偏斜。

本次保持膝轴沿腿外侧约 45 mm 的安装偏置，绕膝轴中心旋转整套设备的安装姿态，使外侧端向下；并做小于 0.001 mm 的轴心投影修正，消除导出舍入偏差。
**没有把 URDF 关节轴强行改成 Z 轴，也没有修改其关节原点或零位。**

校准后检查结果：

| 项目 | 结果 |
|---|---|
| 电机轴心到人体膝轴的距离 | 小于 0.000001 mm |
| 电机轴与人体膝轴夹角 | 小于 0.0001° |
| 左右安装 | 保持镜像对称 |

## 当前安装与合页角度

实际参数保存在 [mounting_preview.json](../mounting_preview.json)。下面的合页角度是预览中的关节状态，不是修改 URDF 的零位。

| 参数 | 右腿 | 左腿 |
|---|---|---|
| 上合页角度 | 1.2058360896° | 1.2058360896° |
| 下合页角度 | −8.25° | −8.25° |
| 相对同侧股骨的安装平移 XYZ（m） | (−0.003639384270, −0.394999687445, 0.045000031898) | (−0.003639384270, −0.394999687445, −0.045000031898) |
| 相对同侧股骨的安装 RPY（°） | (−173.8164333095, −0.0001819583, −0.0000098283) | (6.1835666905, −0.0001819583, 179.9999901717) |

上合页用于将大腿部件的最小骨骼间隙调整到 5 mm。
整套设备转正后，直接保持下合页为 0° 会导致小腿绑缚与胫骨相交，因此下合页调整为 −8.25°，补偿整体旋转并恢复小腿贴合。
电机及小腿相对于校准前的姿态因此有变化；在当前安装姿态下，单独调整上合页不会额外移动电机、输出连接件或小腿。

## 间隙验证

检查的是实际三角网格表面，而非仅根据碰撞开关或最近顶点判断。

| 检查对象 | 已检查膝屈曲角度 | 最小骨骼间隙 | 表面相交 |
|---|---|---|---|
| 左右大腿部件 `base_link` 对全部人体骨骼网格 | 0°、30°、60°、90° | 约 5.0000 mm，最近为同侧股骨 | 未检测到 |
| 右侧小腿部件 `shank_link` 对全部人体骨骼网格 | 0°、70° | 约 4.9595 mm，最近为 `tibia_r_geom_2` | 未检测到 |

左侧小腿的镜像关系另经测试验证。镜像、同轴性、合页调整和网格间隙等共 10 项测试通过。

这些结果只描述所检查姿态下的骨骼网格间隙，不包含皮肤、软组织、红色肌肉/肌腱路径线的穿戴间隙，也不代表绑带压力或任意运动下的无干涉保证。

报告路径（相对 `urdf/knee_exo_hinge/`）：

- `mujoco/mounting_preview/clearance_report.json`：大腿间隙及轴线对齐。
- `mujoco/mounting_preview/calibrated_lower_clearance.json`：小腿间隙。

## 预览与复查命令

以下命令均在仓库根目录 `/home/meng/myoassist` 执行。

静态安装预览：

```bash
.mrl_venv/bin/python -m tools.preview_knee_hinge_mount
```

膝屈伸动画（0–70°，预设运动，不是训练策略或动力学仿真）：

```bash
.mrl_venv/bin/python -m tools.preview_knee_hinge_mount --animate
```

窗口中：橙色为右腿、蓝色为左腿；青色为合页轴、黄色为电机膝轴、绿色为人体膝轴。
保存安装 JSON 后，窗口会自动重载参数。

校准前的 5 mm 安装参数保存在 [mounting_before_axis_alignment.json](../mounting_before_axis_alignment.json)，对比预览：

```bash
.mrl_venv/bin/python -m tools.preview_knee_hinge_mount \
  --config urdf/knee_exo_hinge/mounting_before_axis_alignment.json
```

生成当前安装截图，或重新检查大腿间隙：

```bash
MUJOCO_GL=egl .mrl_venv/bin/python -m tools.preview_knee_hinge_mount --render
.mrl_venv/bin/python -m tools.check_knee_hinge_clearance
```

截图保存在 `urdf/knee_exo_hinge/mujoco/mounting_preview/`。
修改右侧 URDF 后，重新生成左侧镜像的命令为：

```bash
.mrl_venv/bin/python -m tools.mirror_knee_urdf \
  --source urdf/knee_exo_hinge/urdf/knee_exo_hinge_r.urdf \
  --target urdf/knee_exo_hinge/urdf/knee_exo_hinge_l.urdf
```
