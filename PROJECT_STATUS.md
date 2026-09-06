# Molecular-dynamics project status

更新：2026-09-06

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
- 刚体水分子对称性 Demo 已新增为第六套视频：同一个真实 H₂O 在固定周期盒内只做平移和旋转，Cartesian 坐标变化而 O–H、H–H 距离与 cos(H–O–H) descriptor 保持不变；minimum-image descriptor provenance 和 240 帧 QA 已保存。
- 04 Deep Potential 已按组会 PPT 线稿重排：方形完整 water box、左侧原位 O126 局域圆、右侧放大局域圆和绿色真实邻居线；右栏分为 Cartesian/邻居矩阵与 descriptor 两块。
- 04 Deep Potential 已按组会 PPT 线稿重排：方形完整 water box、左侧原位 O126 局域圆、右侧放大局域圆和绿色真实邻居线；右栏分为 Cartesian/邻居矩阵与 descriptor 两块。

独立物理审查已完成并落盘：`_staging/md-qa/independent_physics_review.md`。本轮已按审查意见修正：DP 水盒固定沿 +x 视角；邻居边改为 MatterVis 原生细圆柱线段，每条边从中心原子 `i` 到真实最小镜像邻居 `j`，不带箭头、不承担力语义；neighbor 视角隐藏化学键，只保留原子和真实 `i–j` 边，中心 O126 用选择环标识；力/速度/位置箭头与中心原子锚定；放大图使用同一快照、保留原始周期盒坐标并按最小镜像展开的局域子结构，避免坐标平移和旧 CPU 后端透明度残影；DP 右栏为真实 O126 局部坐标、邻居距离矩阵和由真实 `r_ij` 计算的描述符；刚体水示例改为完整笛卡尔坐标随平移/旋转变化，而 DeepMD `R_i=[s,sx/r,sy/r,sz/r]` 保持不变。邻居边逐条记录在 `neighbor_edge_provenance.json`，共 83 个 cutoff 邻居，静态放大图显示其中 23 个真实 `j`。

四套视频和两个附加视频已完成全帧 QA：VV 216/216、LJ 216/216、AIMD 360/360、DP 384/384、metadynamics 384/384、刚体对称性 240/240，均 `passed=true`。仍未完成的工作仅包括文章最终 XeLaTeX/PDF 排版与逐页 QA、Bohrium Notebook 全单元运行，以及将最终图版嵌入文章。

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

## 可复核交付物
DP 静态 PPT 主图已另行改为纯矢量 SVG：`render_dpmd_vector_static.py` 直接从同一份真实坐标绘制 +x 方向 water box、局域放大、原子端点邻居边、Cartesian 距离矩阵和 DeepPot-SE `R_i` 环境矩阵；SVG 不含嵌入位图。视频仍使用 MatterVis 原生 3D 帧并通过 `visualize-data` 做全帧 QA。

DP 视频时序已改为固定舞台：水盒、O126、小圆、放大圆和引导线全程原位不动；前半段详细展示 `R_i → shared fitting NN → ε_i → E → −∂E/∂r_i → F_i`，右框显示真实 O126 per-atom energy、总能量和力；后半段快速重复邻居、descriptor、力和 VV 更新。颜色 legend 固定在中间主框底部，右框不承担 legend。

当前输出：

- `four_part_story/figures/01_velocity_verlet.png` 至 `04_deep_potential_md.png`
- `four_part_story/videos/01_velocity_verlet.mp4` 至 `04_deep_potential_md.mp4`
- `four_part_story/_qa/04_dpmd_native/qa_report_strict.json`
- `four_part_story/_qa/04_dpmd_native/_qa/every_frame_qa.json`
- `four_part_story/figures/05_well_tempered_metadynamics.png`
- `four_part_story/videos/05_well_tempered_metadynamics.mp4`
- `four_part_story/_qa/05_metadynamics/qa_report_strict.json`
- `four_part_story/figures/06_rigid_water_descriptor_invariance.png`
- `four_part_story/videos/06_rigid_water_descriptor_invariance.mp4`
- `four_part_story/_qa/06_symmetry_invariance/descriptor_provenance.json`
