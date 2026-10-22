# 算法与部署说明

## 输入与目标

输入为上游模型导出的 metric scene flow / point maps、SAM2 或人工生成的首帧末端 mask、专家 HDF5，以及眼在手外相机到机器人 zero/base 坐标系的外参。输出为可用于离线比较或 IK 位置跟踪的 ARX-like HDF5。

## 点流到末端位移

`run_scene_flow_to_arx_pipeline.py` 读取 mask 内有限点，对每帧计算相对第 0 帧的三维位移，并使用逐轴中位数或截尾均值聚合：

```text
M0 = first_frame_mask > 0
delta_pts[t] = F[t, M0] - F[0, M0]
ee_delta_model[t] = aggregate(delta_pts[t])
```

聚合结果经过移动平均，且第 0 帧强制为零。

## 坐标变换与时间对齐

外参四元数按 `[qx, qy, qz, qw]` 解析。位移向量仅应用旋转：

```text
ee_delta_zero = R_cam2zero @ ee_delta_model
```

不能额外做 OpenCV/OpenGL 翻转。预测位置以专家 `observations/ee_pose[0, :3]` 为绝对起点，再按 `sampled_indices` 线性插值到目标帧数（通常 130）。姿态暂时固定为专家首帧姿态。

## HDF5 与夹爪

核心输出是 `observations/ee_pose`。单序列脚本把关节/动作/夹爪等字段作为兼容占位复制；外部 mask 批处理脚本进一步设置前 `N-1` 帧夹爪闭合、末帧打开。

## TraceGen 与可视化

TraceGen flow 被视为逐步增量，并通过稀疏 tracks 与首帧 mask 选点。其已确认的 optical/CV 到 base 映射为 `base_x=cv_z, base_y=-cv_x, base_z=-cv_y`。pretty 可视化把 RGB-D 点云和 expert/ours/TraceGen 三条轨迹变换到 base 坐标系并输出 PNG/PLY。

## 真机边界

外参只对标定时的机械安装有效；相机或机器人移动后必须重新标定。当前输出姿态与非末端状态含占位数据，未经单位、坐标系、工作空间、速度和碰撞检查，不得直接用于无人监督的真机控制。首次验证必须限速并保留急停与人工监督。
