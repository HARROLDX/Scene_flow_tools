# Scene Flow Tools 项目说明

本规则适用于整个仓库；更具体的目录级规则优先。

项目把外部首帧末端 mask 与 metric scene flow 聚合成末端位移，经 `T_cam2end-zero.json` 变换到 ARX zero/base 坐标系，并输出轨迹 HDF5 和点云可视化。用户入口为根目录 `README.md`，历史服务器细节见 `点流工具包使用说明.md`，当前任务和文件索引见 `Agent_doc/Codex_tasks.md`。

维护时遵守以下约定：

- 区分“外部 SAM2 mask 输入”和仓库内实际能力；仓库目前没有完整 SAM2 推理实现。
- 不对 `vae_metric_*` 额外应用 OpenCV/OpenGL 的 `diag([1, -1, -1])` 翻转。
- 后续帧仍使用首帧固定像素集合，不用逐帧 mask 替换。
- 改动坐标系、外参、单位、HDF5 字段或夹爪策略时，同步 README 和任务记录。
- 静态检查使用 `python -m py_compile *.py`；端到端验证需要仓库外数据，不得把缺数据误报为通过。
- 不提交数据集、模型权重、生成的 HDF5、缓存或包含凭据/私密路径的文件。
