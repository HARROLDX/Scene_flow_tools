# 项目与任务入口

## 项目概览

主要链路：`pipeline_outputs.npz + 首帧 mask + 专家 HDF5 + 相机外参` → 固定 mask 像素的 scene-flow 位移聚合 → zero/base 坐标变换 → 平滑与时间插值 → ARX-like HDF5。TraceGen 脚本提供对比轨迹，pretty 脚本生成 PNG/PLY 点云可视化。

## 阅读与文件索引

- `README.md`：公开安装、输入、运行、输出与限制说明。
- `点流工具包使用说明.md`：原服务器数据布局、坐标约定和历史 smoke test。
- `run_scene_flow_to_arx_pipeline.py`：单序列主入口和 HDF5 写出。
- `batch_external_mask_scene_flow.py`：四类任务 ours 批处理入口。
- `batch_external_mask_tracegen_scene_flow.py`：TraceGen 对比批处理入口。
- `batch_pretty_pointcloud_visualizations.py`：可视化批处理入口。
- `T_cam2end-zero.json`：眼在手外相机到机器人 zero/base 的外参。
- `Agent_doc/Code_of_algorithm.md`：算法、坐标链和真机边界。
- `Agent_doc/Codex_mistakes.md`：重要错误经验和稳定版本记录。

## 当前任务（2026-10-22）

目标：阅读仓库，补充一份可公开使用的中文 `README.md` 并推送到 `origin/main`。

已完成：核对全部脚本职责、输入输出、依赖、坐标链、批处理任务、服务器路径和已知限制；README 明确说明 SAM2/数据/模型不包含在仓库中。

验证：执行 Python 语法检查和 Git diff 检查；仓库不含运行数据，因此本次不做端到端重跑。历史 smoke test 证据来自 `点流工具包使用说明.md`。
