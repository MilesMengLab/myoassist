# 带侧向合页的膝外骨骼安装预览

训练流程现已接入本包的同轴安装。统一输入接口为 `training_interface.json`，
使用 `python -m tools.build_knee_training_model` 生成 `mujoco/training/human_knee_exo.xml`。
完整三关节链和被动合页保留，通过数值 cuff weld 连接人体；当前默认训练配置为
70 维完整观测、26 维外骨骼输入，需要新训练权重。

在仓库根目录运行：

```bash
.mrl_venv/bin/python -m tools.preview_knee_hinge_mount
```

预设膝屈伸动画（0–70°，不是训练策略，也不是动力学仿真）：

```bash
.mrl_venv/bin/python -m tools.preview_knee_hinge_mount --animate
```

导出截图：

```bash
MUJOCO_GL=egl .mrl_venv/bin/python -m tools.preview_knee_hinge_mount --render
```

橙色右腿、蓝色左腿，青色线段表示被动合页轴，黄色表示电机膝轴，绿色表示人体膝轴。
安装参数在 `mounting_preview.json`，保存后交互窗口会自动更新。

## 本次处理

- 原始右侧导出文件备份为 `source/knee_exo_hinge_r_export.urdf`。
  原导出文件有两个同名 `knee_joint`，且重复定义同一对父子刚体；删除其中冗余的
  `fixed` 项，保留 `revolute` 项。没有改动导出的 link 坐标、转轴、质量或惯量。
- `urdf/knee_exo_hinge_l.urdf` 和四个 `meshes/*_L.STL` 从修正后的右侧生成。
  真实镜像 STL 顶点、三角面绕序和法向，并镜像所有关节变换、转轴、质心、惯量。
  镜像平面为 CAD 局部 X=0；配合左右安装旋转，得到人体矢状面上的左右镜像。
- 保留膝轴沿腿外侧约 45 mm 的安装偏置；整套设备绕膝轴中心转动约 6.1836°，
  使电机膝轴与人体膝轴同轴。电机轴心做了小于 0.001 mm 的投影修正以消除导出舍入误差。
  电机与小腿随整体安装姿态转动，不能理解为保留它们校准前的方向。
- 校准前的 5 mm 安装保存在 `mounting_before_axis_alignment.json`。对比预览：

  ```bash
  .mrl_venv/bin/python -m tools.preview_knee_hinge_mount --config urdf/knee_exo_hinge/mounting_before_axis_alignment.json
  ```
- 以上方 `motor_housing_link` 为安装基准，固定电机相对于股骨的位置和朝向，
  通过完整 URDF 正向运动学反算大腿 `base_link` 的位姿。
  上合页左右均为 **1.2058361°**；下合页为 **−8.25°**，用于补偿整体安装旋转并调整贴合，
  避免整套旋转后的小腿/胫骨干涉。
  没有独立平移或缩放大腿/小腿零件。
  改变上合页角度不会改变电机外壳、输出件或小腿的位姿，连接轴保持连续。
- 上合页的 1.2058361° 是安装预览关节角，写在安装 JSON；没有篡改 URDF 的关节原点或零位。

## 间隙检查

```bash
.mrl_venv/bin/python -m tools.check_knee_hinge_clearance
```

报告：`mujoco/mounting_preview/clearance_report.json`。

检查左右大腿部件 `base_link` 的真实三角网格与所有显示的骨骼网格。
在人体膝屈曲 0°、30°、60°、90° 四个姿态，最近的骨骼均为同侧股骨，
最小表面间隙约 **5.0000 mm**，未检测到表面相交。调整上合页前后，电机外壳、输出件、
小腿的位置与旋转变化仅为浮点舍入误差；此处比较的是同一个校准后安装姿态内的上合页调整。

5 mm 目标只针对本次调整的大腿部件与骨骼，不表示固定的电机、输出件、小腿也有
5 mm 间隙。骨骼间隙不包含皮肤、软组织和红色肌腱/肌肉路径线的穿戴间隙；有限姿态
采样也不能保证任意运动都无干涉。

校准后另行检查小腿部件与所有骨骼，在膝屈曲 0° 和 70° 时，最小间隙约
**4.9595 mm**，最近为同侧 `tibia_r_geom_2`，没有检测到表面相交。
报告在 `mujoco/mounting_preview/calibrated_lower_clearance.json`。
左侧镜像及同轴性、零件连续运动的 10 项检查通过。

## 电机轴校准

原始膝关节轴为 `(0, 0.10771, 0.99418)`，位于其 URDF 关节坐标系。
按校准前的电机安装姿态合成人体后，轴线位置几乎重合，但方向与人体膝轴相差约
**6.1836°**，从腿内侧朝外侧看为外侧向上。现在绕轴心旋转整套设备的安装姿态，
使外侧端向下，保持 URDF 轴定义及零件内部连接不变。
当前轴线位置误差小于 **0.000001 mm**，方向误差小于 **0.0001°**。
上述预览只展示几何安装。训练构建器现已另行保留三关节链，并用数值 weld 形成
人体绑缚闭环；它不包含真实皮肤、软组织或绑带压力模型。

重新生成左侧镜像：

```bash
.mrl_venv/bin/python -m tools.mirror_knee_urdf \
  --source urdf/knee_exo_hinge/urdf/knee_exo_hinge_r.urdf \
  --target urdf/knee_exo_hinge/urdf/knee_exo_hinge_l.urdf
```
