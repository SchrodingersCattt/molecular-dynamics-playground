# Molecular-dynamics project status

更新：2026-09-05

## 已完成并留存

- Integrator Notebook 已修正为 Explicit Euler、Symplectic Euler、Leapfrog/Verlet、Velocity Verlet、RK2 和固定步长 Classical RK4；时间网格、能量误差和二阶收敛结果已保存。
- 两份 Integrator Notebook 已于 2026-09-05 重新执行，本地脚本无异常，比较图 PNG/PDF 已同步更新。
- 四套视觉交付物已有独立 A4 静态图和 16:9 视频入口：Velocity Verlet、TIP3P/Lennard–Jones、水二聚体 RHF–SCF、64 水 Deep Potential MD。
- AIMD 已保存多个离子步、每个离子步的 SCF 密度/残差、固定分子平面网格和位置/速度/力阶段素材；RHF 画面把等值线随真实密度收敛呈现为由粗到清晰。
- DPMD 已保存 64 水周期盒、O126 中心原子、83 个 6 Å 内最小镜像邻居、同一快照的 DP 能量/力和可复现的冻结力 Velocity–Verlet 步。
- DP 截断邻域已改为 MatterVis 原生世界坐标球面；球心与 O126 对齐，投影保持正圆，采用固定斜视相机、方向光、前后半球透明度和原生 MIC 向量。对应素材位于 `four_part_story/_qa/04_dpmd_native/mattervis_v3/`。
- `visualize-data --strict` 已用于静态图；DP 视频保留 384/384 帧逐帧 QA 报告，尺寸为 1920×1080、24 fps。
- MatterVis 已增加原生 overlay metadata 传递、CPU 球面方向光/高光、零透明度原子/键隐藏和对应测试；测试文件为 `MatterVis/tests/test_native_overlay_metadata.py`。
- 中文读书笔记已有 LaTeX 源、参考文献、PDF 和渲染页；历史、Euler/Verlet、SCF、Deep Potential、采样章节已加入具体数字例子。
- well-tempered metadynamics Demo 已新增为第五套独立静态图和 16 秒视频；一维双阱、Langevin 初期驻留、well-tempered Gaussian hills 和自由能恢复均有素材，静态 strict QA 与 384 帧视频 QA 已通过。

## 当前推送内容

- `molecular-dynamics-playground`：全部交付已直接推送到 `main`，当前远端提交为 `3d0b50f`。此前 PR [#1](https://github.com/SchrodingersCattt/molecular-dynamics-playground/pull/1) 已被 GitHub 标记为 MERGED；后续按要求直接推 `main`。
- `molecular-dynamics-playground` 视觉分支也保留在 `visual/final-20260905`，最新提交为 `3b197f5`，包含完整 MatterVis 素材、第五套 metadynamics Demo 和 QA 文件。
- `MatterVis`：`fix/cpu-vector-style-parity` 已推送至 merge 提交 `199f69b`，包含功能提交 `32e5380`，PR 为 [#133](https://github.com/SchrodingersCattt/MatterVis/pull/133)。所有提交均为追加式，没有 amend、rebase、squash 或 force-push。

## 仍需完成的工作

- 对四套视频做最终统一逐帧复核，尤其是 AIMD 的 SCF 闪烁、离子步停顿和 DP 的球内/球外淡化；当前 DP 最终版本已通过 384/384 帧检查，其余版本的最后一次导出仍需再做同一轮汇总审阅。
- MatterVis 到 `main` 的新 PR 已创建；已有 PR #117 的冲突不在本轮历史中改写，后续由 PR #133 继续审阅。
- 在可用的 MiKTeX/XeLaTeX 环境重新编译读书笔记，重新生成 PDF、逐页回渲和最终排版 QA；现有源稿和上一版 PDF 已保存，但修订后编译曾受本机 MiKTeX 权限错误影响。
- 将 metadynamics 图和视频纳入文章最终排版，并补充 umbrella sampling 的文字示意与采样误差说明。
- 在 Bohrium Notebook 上用修订稿运行全部单元并保存输出；这一步依赖公开 Notebook 的替换和计算节点运行。

## 可复核证据位置

- 视觉说明：`four_part_story/README.md`
- DP 数据与 QA：`four_part_story/_qa/04_dpmd_native/`
- 文章源稿与 PDF：`report/md_reading_notes.tex`、`report/output/pdf/md_reading_notes.pdf`
- 文章修订日志：`report/_qa_revision_log.md`
- MatterVis API 记录：`MatterVis/docs/agents/scene_api.md`
