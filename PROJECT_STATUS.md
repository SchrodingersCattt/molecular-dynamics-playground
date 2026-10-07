# Molecular-dynamics project status

更新：2026-10-07

## 2026-10-07：TNT C2–NO2 replacement in progress

- TNT topology now comes from the RDKit 2,4,6-trinitrotoluene SMILES with an explicit atom-order map; the selected breaking site is C2–N2. The previous hand-built coordinates could trigger a spurious O–O bond in MatterVis and are no longer used.
- `scripts/md_visuals/inspect_tnt_structure.py` runs the mandatory `mat-vis inspect` plus CPU `ball_stick` render preflight. The checked local reference has formula C7H5N3O6, 21 atoms, 21 perceived bonds, no O–O bond, and C2–N2 = 1.46 Å.
- TNT 03b demo data currently save fixed-grid 3-D alpha/beta density in `product/data/uks_tnt_reaction_density3d.npz`; the scientific `backend=pyscf_uks` dataset still requires the Bohrium run.

## 2026-10-07（第二轮）：PPT 字号、统一 r/v/a 配色、01/02b/03/03b/04/04b 修订

- 全局：视频按整页宽 16:9 PPT（13.333 in）换算字号，`common.VideoFigure` 在绘制时把所有文字放大 `PPT_TEXT_SCALE ≈ 1.44`，因此 `FONT_SIZES` 就是 PPT 中的磅值：14（注释/图例/刻度）、16（标签/公式）、18（面板标题）、24（循环 r/v/a 符号）。全部去掉粗体（矢量符号除外），字体 Arial。配色固定：r/位移 蓝 `#2F6FB3`、v 紫 `#7A4FB8`、a/F 橙 `#E07A1F`，能量 青 `#1F7A7A`；VV 循环节点、弧、箭头和场景箭头共用这套颜色，各视频场景右下角加箭头图例。循环的 acceleration/velocity 标签移到圆环下方，不再被弧遮挡。
- 01：球从第一帧起全部就位；速度阶段改为“三张直方图的点 → 每球一支合成速度箭头（在直方图上方汇成星形）→ 所有箭头并行平移到各球”；去掉扣除质心动量和温度缩放两步；箭头缩短；箭头切换不做淡入淡出（只有球的运动保留插值）。
- 02b：右上 BO 曲线加图例（Cl=O / Cl–OH (breaking)），去掉上、右边框，坐标轴改黑色；右下小环改为显式能量式（BO、E_bond、E_over、E_angle、E_Coul、F=−∂E/∂r），随阶段高亮。断裂键改深墨色，能量项提示（配位圈、键角弧、δ±）统一青色。
- 03 / 03b：残差图改为上/右无框、黑轴、14 pt 刻度、纵轴 "residual"，避免与 SCF 环标签相撞；SCF 状态行简化为 "SCF iteration NN / NN"；03 速度/位移箭头缩短约 30% 且相机上移，顶部 H 的箭头不再被截；03b 渲染画幅 1500→1650 px 宽（正交尺度按高度，分子大小不变），离去 OH 的外层等值线不再在右侧被截；α/β density 图例移到右下，只在 SCF 阶段显示。
- 04 / 04b：放大镜源结构保留完整水分子（按 O 归属展开），化学视图画键，r_c 外原子淡化、键随较淡端点淡化；邻居视图仍为孤立原子。原位截断圆、引导线和放大镜里的截断球从第一帧起同时出现。标题缩短、步号统一为 "Simulation step NN (t fs)"；右栏去掉维度/padding/并行气泡等小字，ε_i 与 E 左对齐；F 返回路径改为一条实线：从 E 文字正中向下、沿底边向左、沿 A|B 间隙向上、从右侧进入 a 节点，力阶段橙色生长，其余时间灰色。配色收敛为：结构/描述符 navy，fitting/ε/E 青，力与反向传播 橙，速度 紫，位移 蓝。
- 第一轮备份仍在 `_tmp/backup_20261007/`。

## 2026-10-07：01 / 02b / 04 / 04b 打磨（样式与时序，不改布局）

- 01：去掉底面、落点线和坐标系；相机绕视线自动选择 roll，使球堆横向铺满中间 panel（球径约 +40%），渲染画幅改为 1700×1000。r 阶段：球按高度逐个出现，带中心点；快扫的每个 r 三分之一先显示蓝色位移箭头再移动。v 阶段的速度分配改为逐球进行：三张直方图上该球的三个抽样点加圈，目标球加虚线圈，三点飞入后该球箭头长出；前两个球慢放（优先无遮挡的球），其余 0.3 s 错开。时长 21 s → 22 s。`Simulation step` 移到中间 panel 左下角最底部。
- 02b：原子与键改为 MatterVis 原生 ball-and-stick（与 03/03b 相同的 atom scale 0.90、bond radius 0.102）；只有断裂的 Cl–O(H) 键单独着色为橙色，透明度和粗细随键级变化，并在超出 MatterVis 成键距离后保留到 BO < 0.05。`mattervis_story.render_structure` 新增 `bond_styles` 参数。含义不明的电荷小圆点改为 `δ+`/`δ−` 标签，放在每个原子成键方向的最大空隙里，避开配位虚线圈。
- 04 / 04b：去掉 j₁–j₃ 的放大圆标注、盒子下方的列表、矩阵上的行框和 `j_{1-3}` 标签。快扫阶段改为两次连续擦除：先沿行块从左到右（embedding），再沿右栏从上到下（Σ_j → D → fitting → ε → E），各约 0.44 s，带细扫描线；力、加速度、速度、位置各留出可见时长。
- 原脚本和视频备份在 `_tmp/backup_20261007/`。

## 2026-10-06：01 塑料球 Velocity Verlet 与 02b 示意 ReaxFF

- 01 重做：9 个相同的硬塑料质感球（LJ，100 K，dt = 80 fs，无溶剂）替代 H2O。数据生成 `scripts/build_box/generate_plastic_vv.py` → `product/data/vv_plastic_balls.npz/.json`；渲染 `scripts/md_visuals/render_velocity_verlet.py`，左侧共享 VV 循环，中间 MatterVis 球堆 + 细坐标系/落点线，右侧三张 v_x/v_y/v_z 高斯直方图。初速度按 LAMMPS `velocity create dist gaussian mom yes rot no` 流程：三个分量各抽一个数、扣除质心速度、整体缩放到目标 T；随后一次完整 VV 步和六步快速循环。箭头改为细轴小头，右侧不含文字框。
- 02b 新增：`scripts/build_box/generate_reaxff_hclo4.py` 在 03b 的 HClO4 几何和初速度上运行示意性 ReaxFF 型能量（键级、键能、过配位惩罚、BO 加权键角、EEM 电荷、核心排斥），torch autograd 力经有限差分校验，velocity Verlet 积分；渲染 `scripts/md_visuals/render_reaxff.py`，布局同 03b（左 VV 循环、中间分子 + 键级管、右上键级-距离曲线、右下 r_ij→BO→E→F 小环 + 三个卫星）。画面保持 `schematic` 标注；它不是已发表的 ReaxFF 参数集，也没有 LAMMPS 运行。
- 合同：`product/qa/01_velocity_verlet/figure_contract.md`（已改写）、`product/qa/02b_reaxff/figure_contract.md`（新增）。

## 2026-10-06：03b UKS 反应型 AIMD 管线

- 新增 `scripts/run_md/engine_uks.py`、`scripts/build_box/generate_uks_hclo4.py` 和基于 `render_aimd_scf.py` 的 `scripts/md_visuals/render_uks_aimd.py`，对应 `03b_uks_reaction` 的 MatterVis 静态图、30 秒 16:5 视频、数据 manifest 和全帧 QA；`render_uks_reaction.py` 保留为兼容入口并转发到同一渲染器。
- 反应叙事固定为 `HClO4 → ·OH + ·ClO3` 的 Cl–O(H) 同裂解；轨迹使用 OH/ClO3 质量加权反向初速度，记录 Cl–O 距离、O–H 距离、spin metric、⟨S²⟩ 和 UKS SCF 接口字段。
- 当前 Windows 环境没有可直接使用的 PySCF wheel，源代码构建还缺少 C/C++ 编译器；因此已生成 `backend=analytic_demo` 的明确标注演示数据。03b 的 MatterVis 视频已按 03 AIMD 的 30 秒时间轴完成快速导出和代表性帧检查；真实 UKS 结果需在安装平台支持的 PySCF 后运行同一生成脚本（不带 `--demo`）。

## 2026-10-02：DeepMD 与 DPA4C 端到端重做

- 新渲染器 `scripts/md_visuals/render_nnmd_end_to_end.py` 同时产出 `04_deep_potential_md` 与 `04_4c_dpa4c` 的 30 s 视频和 A4 静态图，排版与 AIMD 一致：左侧 VV 循环、中间真实 64 水盒与 O126 局域放大、右上真实 E(step)、右下 E→∂E/∂r→F→a→Δt 信息链，并由底部回到循环的 a 节点。两套视频只有“力的提供者”不同。
- 所有物理量直接画在真实原子上：83 条 O126 邻居边、三条最近邻的真实 r 与 DeepPot-SE 行 / DPA4C 单位向量、按 ε_j 偏差着色的原子、r_c 内每个原子的力/加速度/半步速度/位移箭头；箭头比例和颜色范围由数据决定并写入 `asset_manifest.json`。
- DPA4C 轨迹为 Bohrium 作业 20808156 的真实模型输出（`DPA4C-Neo-OMat24-v20260819.pt`，deepmd-kit 3.2.0），与 DeepMD 轨迹使用同一水盒、同一初速度与步长；数据在 `product/data/dpa4c_water_box_trajectory.npz/.json`，作业脚本与日志在 `product/qa/04_4c/bohr_live_water_v2/`。
- 旧的深色版本 `render_nnmd.py` 及其素材已删除。说明见 `docs/04_nnmd_end_to_end.md`、`docs/04_4c_dpa4c.md`。

## 已完成并留存

- Integrator Notebook 已修正为 Explicit Euler、Symplectic Euler、Leapfrog/Verlet、Velocity Verlet、RK2 和固定步长 Classical RK4；时间网格、能量误差和二阶收敛结果已保存。
- 两份 Integrator Notebook 已于 2026-09-05 重新执行，本地脚本无异常，比较图 PNG/PDF 已同步更新。
- 视觉交付物已有独立 A4 静态图和 16:5 视频入口；视频统一为 1920×600、Arial 16–18 pt。
- AIMD 已保存多个离子步、每个离子步的 SCF 密度/残差、固定分子平面网格和位置/速度/力阶段素材；RHF 画面把等值线随真实密度收敛呈现为由粗到清晰。
- DPMD 已保存 64 水周期盒、O126 中心原子、83 个 6 Å 内最小镜像邻居、同一快照的 DP 能量/力和可复现的冻结力 Velocity–Verlet 步。
- DP 截断邻域已改为 MatterVis 原生世界坐标球面；球心与 O126 对齐，投影保持正圆，采用固定斜视相机、方向光、前后半球透明度和原生 MIC 向量。对应素材位于 `product/qa/04_dpmd_native/mattervis_v3/`。
- `visualize-data --strict` 已用于静态图；视频逐帧 QA 同时检查 16:5、Arial、字号下限和上下边缘留白。
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

- 视觉说明：`docs/four_part_story.md`
- 活动脚本：`scripts/md_visuals/`、`scripts/build_box/`、`scripts/run_md/`、`scripts/submit_calculation/`。
- DP 数据与 QA：`product/qa/04_dpmd_native/`
- 文章源稿与 PDF：`report/md_reading_notes.tex`、`report/output/pdf/md_reading_notes.pdf`
- 文章修订日志：`report/_qa_revision_log.md`
- MatterVis API 记录：`MatterVis/docs/agents/scene_api.md`

## 可复核交付物
DP 静态 PPT 主图已另行改为纯矢量 SVG：`render_dpmd_vector_static.py` 直接从同一份真实坐标绘制 +x 方向 water box、局域放大、原子端点邻居边、Cartesian 距离矩阵和 DeepPot-SE `R_i` 环境矩阵；SVG 不含嵌入位图。视频仍使用 MatterVis 原生 3D 帧并通过 `visualize-data` 做全帧 QA。

DP 视频时序已改为固定舞台：水盒、O126、小圆、放大圆和引导线全程原位不动；前半段详细展示 `R_i → shared fitting NN → ε_i → E → −∂E/∂r_i → F_i`，右框显示真实 O126 per-atom energy、总能量和力；后半段快速重复邻居、descriptor、力和 VV 更新。颜色 legend 固定在中间主框底部，右框不承担 legend。

当前输出：

- `product/figures/03b_uks_reaction.png` / `.svg`
- `product/videos/03b_uks_reaction.mp4`
- `product/data/uks_hclo4_reaction.npz` / `.json`
- `product/qa/03b_uks_reaction/qa_report_strict.json`
- `product/figures/01_velocity_verlet.png` 至 `04_deep_potential_md.png`
- `product/videos/01_velocity_verlet.mp4` 至 `04_deep_potential_md.mp4`
- `product/qa/04_dpmd_native/qa_report_strict.json`
- `product/qa/04_dpmd_native/_qa/every_frame_qa.json`
- `product/figures/05_well_tempered_metadynamics.png`
- `product/videos/05_well_tempered_metadynamics.mp4`
- `product/qa/05_metadynamics/qa_report_strict.json`
- `product/figures/06_rigid_water_descriptor_invariance.png`
- `product/videos/06_rigid_water_descriptor_invariance.mp4`
- `product/qa/06_symmetry_invariance/descriptor_provenance.json`


