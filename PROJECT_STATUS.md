# MD visual delivery status

更新：2026-09-06

已完成：四套独立静态图和视频脚本；MatterVis 原生结构、双色键、周期盒、世界坐标向量和 DP 6 Å 邻域球；AIMD/RHF 多离子步与 SCF 密度素材；DP 64 水盒的真实能量、力、中心原子和最小镜像邻居；静态图严格 QA；DP 视频 384/384 帧 QA。完整素材、sidecar 和报告均在 `four_part_story/_qa/`。

本轮重点修订：DP 静态 PPT 图严格采用正方形周期水盒，局部 O126 的正圆 r_c 邻域直接覆盖水盒中心；右栏上半显示真实局部笛卡尔坐标和 O126 邻居距离矩阵，下半显示由真实 r_ij 计算的 radial descriptor。动画随后展示 descriptor → shared NN → Σ ε_i → E → F。数据证据写入 `_qa/04_dpmd_native/descriptor_provenance.json`；球面使用 MatterVis 原生方向光和固定相机。

独立物理审查已完成并落盘：`_staging/md-qa/independent_physics_review.md`。本轮已按审查意见修正：DP 水盒固定沿 +x 视角；邻居边改为 MatterVis 原生细圆柱线段，每条边从中心原子 `i` 到真实最小镜像邻居 `j`，不带箭头、不承担力语义；neighbor 视角隐藏化学键，只保留原子和真实 `i–j` 边，中心 O126 用选择环标识；力/速度/位置箭头与中心原子锚定；放大图使用同一快照、保留原始周期盒坐标并按最小镜像展开的局域子结构，避免坐标平移和旧 CPU 后端透明度残影；DP 右栏为真实 O126 局部坐标、邻居距离矩阵和由真实 `r_ij` 计算的描述符；刚体水示例改为完整笛卡尔坐标随平移/旋转变化，而 DeepMD `R_i=[s,sx/r,sy/r,sz/r]` 保持不变。邻居边逐条记录在 `neighbor_edge_provenance.json`，共 83 个 cutoff 邻居，静态放大图显示其中 23 个真实 `j`。

四套视频和两个附加视频已完成全帧 QA：VV 216/216、LJ 216/216、AIMD 360/360、DP 384/384、metadynamics 384/384、刚体对称性 240/240，均 `passed=true`。仍未完成的工作仅包括文章最终 XeLaTeX/PDF 排版与逐页 QA、Bohrium Notebook 全单元运行，以及将最终图版嵌入文章。

DP 静态 PPT 主图已另行改为纯矢量 SVG：`render_dpmd_vector_static.py` 直接从同一份真实坐标绘制 +x 方向 water box、局域放大、原子端点邻居边、Cartesian 距离矩阵和 DeepPot-SE `R_i` 环境矩阵；SVG 不含嵌入位图。视频仍使用 MatterVis 原生 3D 帧并通过 `visualize-data` 做全帧 QA。

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
