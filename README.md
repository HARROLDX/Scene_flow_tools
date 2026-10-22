# Scene Flow Tools

面向 **眼在手外（eye-to-hand）相机 + ARX 机械臂** 的末端轨迹生成工具包。

本项目使用 SAM2 或人工标注得到的首帧机械臂末端 mask，在预测的 metric scene flow（点流）中固定选取末端像素，并聚合这些点随时间的三维位移；随后利用相机到机器人 zero/base 坐标系的外参完成坐标变换，最终生成可供机器人 IK 跟踪或离线评估的 ARX-like HDF5 轨迹，同时支持与专家轨迹、TraceGen 轨迹进行点云可视化对比。

> [!IMPORTANT]
> 当前仓库包含的是点流轨迹后处理工具，不包含 SAM2 权重、完整 SAM2 推理代码、上游 scene-flow 模型或示例数据。批处理脚本默认读取已生成的首帧 mask 和 `pipeline_outputs.npz`。

## 工作流程

```text
外部 RGB / RGB-D 相机（眼在手外）
            │
            ├── SAM2 / 人工标注 ──> 首帧末端 mask M₀
            │
            └── scene-flow 模型 ──> metric scene flow Fₜ
                                      │
                                      ▼
                    固定 M₀ 内像素并聚合三维位移
                                      │
                                      ▼
                   相机坐标系 ──外参──> 机器人 zero/base 坐标系
                                      │
                                      ▼
                   平滑、首帧对齐、插值到 130 帧
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
                  ARX-like HDF5               PNG / PLY 可视化
```

对首帧 mask 中的有效像素，末端位移估计为：

```python
delta_pts[t] = F[t, M0] - F[0, M0]
ee_delta[t] = median(delta_pts[t])
ee_delta_zero[t] = R_cam2zero @ ee_delta[t]
```

这里始终使用同一个首帧 mask，不会在后续帧重新取 mask。`vae_metric_*` 的坐标约定已与本项目外参匹配，因此不要再额外乘 `diag([1, -1, -1])` 做 OpenCV/OpenGL 翻转。

## 仓库结构

| 文件 | 作用 |
| --- | --- |
| `run_scene_flow_to_arx_pipeline.py` | 单条 scene-flow 轨迹转换：NPZ + 首帧 mask → ARX-like HDF5 |
| `batch_external_mask_scene_flow.py` | 使用外部首帧 mask 批量处理四类任务 |
| `batch_external_mask_tracegen_scene_flow.py` | 批量生成 TraceGen 对比轨迹 |
| `batch_pretty_pointcloud_visualizations.py` | 批量生成 RGB 点云轨迹图和 PLY |
| `make_pretty_pointcloud_trajectory.py` | 单条序列的点云轨迹可视化 |
| `batch_cup_tracegen_scene_flow.py` | TraceGen 轨迹处理和可视化公共实现 |
| `T_cam2end-zero.json` | 相机坐标系到机器人 zero/base 坐标系的外参 |
| `点流工具包使用说明.md` | 已验证服务器环境、数据布局和历史运行说明 |

## 环境依赖

- Python 3.9+
- NumPy
- h5py
- Pillow
- OpenCV (`opencv-python`)
- Matplotlib

可以在独立虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy h5py pillow opencv-python matplotlib
```

仓库暂未提供锁定版本的依赖文件；如需复现实验服务器上的已验证环境，请参考 [`点流工具包使用说明.md`](./点流工具包使用说明.md)。

## 输入数据

### 1. Scene-flow NPZ

单条轨迹转换至少需要：

- `vae_metric_scene_flow`：metric scene flow，批处理脚本使用此键；
- `vae_metric_point_maps`：metric point maps；
- `sampled_indices`：模型帧对应的原始时间索引。

`run_scene_flow_to_arx_pipeline.py` 的默认键是旧名称 `vae_scene_flow` 和 `vae_point_maps`，可通过 `--flow_key`、`--point_key` 显式指定。批处理脚本已自动指定 metric 键。

### 2. 首帧末端 mask

mask 是单通道图像，非零像素表示机械臂末端区域。批处理目录约定为：

```text
<mask_root>/<task>/<sequence_id>_first_frame_mask.png
```

例如：

```text
ARX_masks/green_tray/0_first_frame_mask.png
```

mask 可由 SAM2 生成，也可人工标注。若尺寸与 RGB 不同，批处理脚本会使用最近邻插值调整尺寸。

### 3. 外参

`T_cam2end-zero.json` 使用以下格式：

```json
{
  "position": [0.0, 0.0, 0.0],
  "orientation": [0.0, 0.0, 0.0, 1.0],
  "rpy": [[0.0, 0.0, 0.0]]
}
```

轨迹位移只使用四元数对应的旋转矩阵；平移项会被载入并记录，但位移向量不需要叠加平移。

### 4. 专家 HDF5

专家轨迹用于提供首帧末端绝对位置、姿态和输出模板。核心数据集为：

```text
observations/ee_pose
```

`qpos`、`qvel`、`effort`、`action` 等非末端轨迹字段当前只是从专家首帧复制的占位值。

## 快速开始：处理单条轨迹

建议显式传入所有路径，避免使用脚本中面向原实验服务器的默认路径：

```bash
python run_scene_flow_to_arx_pipeline.py \
  --sample_dir /path/to/sample \
  --pipeline_npz /path/to/sample/pipeline_outputs.npz \
  --rgb_dir /path/to/sample/per_frame_exports/rgb_pred \
  --mask0 /path/to/first_frame_mask.png \
  --expert_hdf5 /path/to/episode_0_ee_pose_arx.hdf5 \
  --cam2zero_json ./T_cam2end-zero.json \
  --flow_key vae_metric_scene_flow \
  --point_key vae_metric_point_maps \
  --scale 1.0 \
  --expert_cut 130 \
  --out_dir /path/to/output \
  --skip_sam_if_missing
```

主要参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--scale` | 点流位移缩放系数 | `0.6` |
| `--expert_cut` | 输出/比较帧数上限 | `130` |
| `--smooth_window` | 移动平均窗口 | `5` |
| `--agg` | mask 内位移聚合方法：`median` 或 `trimmed_mean` | `median` |
| `--trim_frac` | `trimmed_mean` 丢弃离群点的比例 | `0.20` |

> `--run_sam_if_missing` 会尝试调用工作目录中的 `segment_ee_with_sam2.py`，但该文件目前不在本仓库中。除非你另行提供该脚本，否则请准备好 mask 并使用 `--skip_sam_if_missing`。

## 批量处理

当前批处理配置包含：

| 任务 | 序列数 |
| --- | ---: |
| `grasp_cube` | 10 |
| `grasp_cup` | 10 |
| `green_tray` | 5 |
| `yellow_tray` | 5 |

批量脚本中的数据根目录目前是原实验服务器的硬编码路径。如在其他机器上运行，请先修改各脚本顶部的 `TASKS` 配置；其中 mask 根目录可通过 `--mask_root` 覆盖。

```bash
# 1. 生成 ours 轨迹
python batch_external_mask_scene_flow.py --tasks green_tray

# 2. 生成 TraceGen 对比轨迹
python batch_external_mask_tracegen_scene_flow.py --tasks green_tray

# 3. 生成点云轨迹可视化
python batch_pretty_pointcloud_visualizations.py \
  --tasks green_tray \
  --start 0 \
  --end 0
```

推荐按上述顺序运行，确保可视化读取的是最新 HDF5。

## 输出

单条 ours 序列的主要输出包括：

```text
pred_ee_pose_arx_like_130.hdf5
pred_ee_pose_arx_like_130_gripper_closed_then_open.hdf5  # 批处理生成
scene_flow_ee_pipeline_arrays.npz
expert_vs_pred_scaled_compare_0_129.csv
frame0000_ee_mask_overlay.png
summary.json
```

批处理最终 HDF5 的夹爪策略为：前 `N-1` 帧闭合（`0.0`），最后一帧打开（`5.0`）。真正用于 IK 跟踪的核心数据为：

```text
observations/ee_pose[:, :3]
```

点云可视化还会生成：

```text
pretty_pointcloud_trajectory_3d.png
pretty_pointcloud_trajectory_top.png
pretty_pointcloud_trajectory_side.png
pretty_pointcloud_trajectory.ply
pretty_pointcloud_trajectory_summary.json
```

轨迹配色为 expert（浅蓝）、ours（红）和 TraceGen（橙）。

## 外参与真机安全

- `T_cam2end-zero.json` 仅适用于生成它的相机—机器人安装关系；移动相机或机器人后必须重新进行眼在手外标定。
- 运行前检查单位、坐标轴方向、外参方向和机器人 zero/base 定义。
- 输出中的姿态和关节相关字段目前含占位数据，不应未经检查直接用于真机全状态控制。
- 首次真机验证应限速、限制工作空间，并保持急停和人工监督。

## 已知限制

- 数据根目录和任务表仍针对原实验服务器，尚未抽象为统一配置文件。
- 仓库不包含 SAM2、上游 scene-flow/TraceGen 推理、数据集和模型权重。
- 输出姿态固定为专家首帧的 `roll/pitch/yaw`，没有从点流估计旋转。
- 单条转换脚本生成的夹爪值沿用专家首帧；“末帧打开”策略由批处理脚本后处理得到。
- 当前仓库没有自动化测试和可直接运行的最小示例数据。

## 验证状态

历史说明记录了 `green_tray / sequence 0` 的 smoke test，可成功生成 ours HDF5、TraceGen HDF5、PNG 和 PLY。由于本仓库未附带数据，本地只能进行静态/语法检查；完整复现需要准备原始 NPZ、mask、专家 HDF5 和 TraceGen 输出。

更详细的服务器数据路径、坐标约定和历史验证结果见 [`点流工具包使用说明.md`](./点流工具包使用说明.md)。
